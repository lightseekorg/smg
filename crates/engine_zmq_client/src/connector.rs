// SPDX-License-Identifier: Apache-2.0
//
// The connector: the frontend-side request/response layer over a
// [`ConnectedTransport`]. Adapted from the Apache-2.0 reference
// `vllm-engine-core-client` (vllm-project/vllm) client, scoped to SMG's needs:
//
// - No client-side DP load balancing. SMG routing picks the rank and stamps
//   `data_parallel_rank`; the connector consumes the piggybacked per-rank
//   `scheduler_stats` as the load signal.
// - No DP coordinator / wave protocol yet (single shared transport, DP=1).
// - No utility RPC (deferred).

use std::{collections::HashMap, sync::Arc, time::Duration};

use bytes::Bytes;
use futures::Stream;
use parking_lot::Mutex;
use tokio::{sync::mpsc, task::JoinHandle};
use tracing::{trace, warn};
use zeromq::RouterSendHalf;

use crate::{
    error::{Error, Result},
    protocol::{
        tokenspeed::TokenSpeedProtocol, vllm::VllmProtocol, EngineBatch, EngineLoad, EngineOutput,
        EngineProtocol,
    },
    transport::{run_output_loop, send_message, ConnectedEngine, ConnectedTransport, EngineId},
};

/// The vLLM EngineCore connector (the original engine surface).
pub type EngineCoreClient = Client<VllmProtocol>;
/// The per-request output stream for the vLLM EngineCore connector.
pub type EngineCoreStream = RequestStream<VllmProtocol>;
/// The TokenSpeed connector.
pub type TokenSpeedClient = Client<TokenSpeedProtocol>;
/// The per-request output stream for the TokenSpeed connector.
pub type TokenSpeedStream = RequestStream<TokenSpeedProtocol>;

type OutputSender<O> = mpsc::UnboundedSender<Result<O>>;
type OutputReceiver<O> = mpsc::UnboundedReceiver<Result<O>>;

/// Routes engine outputs to per-request streams and tracks in-flight requests.
/// Keyed by `request_id`; the value is that request's output channel sender.
struct RequestRegistry<O> {
    closed: bool,
    requests: HashMap<String, OutputSender<O>>,
}

impl<O> Default for RequestRegistry<O> {
    fn default() -> Self {
        Self {
            closed: false,
            requests: HashMap::new(),
        }
    }
}

impl<O: EngineOutput> RequestRegistry<O> {
    /// Register a new request, returning the receiver for its output stream.
    fn register(&mut self, request_id: String) -> Result<OutputReceiver<O>> {
        if self.closed {
            return Err(Error::ClientClosed {
                message: "client is shutting down".to_string(),
            });
        }
        if self.requests.contains_key(&request_id) {
            return Err(Error::DuplicateRequestId { request_id });
        }
        let (sender, receiver) = mpsc::unbounded_channel();
        self.requests.insert(request_id, sender);
        Ok(receiver)
    }

    /// Deliver one output to its request stream; drop the entry when terminal.
    fn route(&mut self, output: O) {
        let request_id = output.request_id().to_string();
        let Some(sender) = self.requests.get(&request_id) else {
            trace!(%request_id, "output for unknown/finished request; dropping");
            return;
        };
        let finished = output.finished();
        if sender.send(Ok(output)).is_err() || finished {
            // Receiver gone (stream dropped) or request finished — stop tracking.
            self.requests.remove(&request_id);
        }
    }

    /// Remove finished/aborted request ids reported out-of-band by the engine.
    fn remove_all<'a>(&mut self, request_ids: impl IntoIterator<Item = &'a String>) {
        for request_id in request_ids {
            self.requests.remove(request_id);
        }
    }

    /// Fail every in-flight request with a shared error and close the registry.
    fn fail_all(&mut self, error: Arc<Error>) {
        self.closed = true;
        for (_, sender) in self.requests.drain() {
            let _ = sender.send(Err(Error::Shared(error.clone())));
        }
    }
}

struct ClientInner<P: EngineProtocol> {
    /// Shared input ROUTER send half (serialized across concurrent submits).
    input_send: tokio::sync::Mutex<RouterSendHalf>,
    engines: Vec<ConnectedEngine>,
    registry: Mutex<RequestRegistry<P::Output>>,
    /// Latest per-rank load, keyed by engine index. SMG's DP load signal.
    load: Mutex<HashMap<u32, EngineLoad>>,
    /// Auto-abort channel fed by dropped streams.
    abort_tx: mpsc::UnboundedSender<(EngineId, String)>,
}

impl<P: EngineProtocol> ClientInner<P> {
    /// Pick the engine for a request: by `data_parallel_rank` if set, else the
    /// sole engine. SMG routing is authoritative for rank selection. The rank is
    /// matched against the engine's ZMQ index (Python `EngineCoreProc`'s 2-byte
    /// identity), which is version-independent — unlike the `data_parallel_rank`
    /// field, which not every vLLM version sends in `EngineCoreReadyResponse`.
    fn select_engine(&self, data_parallel_rank: Option<u32>) -> Result<EngineId> {
        match data_parallel_rank {
            Some(rank) => self
                .engines
                .iter()
                .find(|engine| engine.engine_id.engine_index() == Some(rank))
                .map(|engine| engine.engine_id.clone())
                .ok_or(Error::InvalidDataParallelRank {
                    rank,
                    num_engines: self.engines.len() as u32,
                }),
            None => self
                .engines
                .first()
                .map(|engine| engine.engine_id.clone())
                .ok_or(Error::ClientClosed {
                    message: "no engines connected".to_string(),
                }),
        }
    }

    /// Send an encoded request frame to one engine over the shared input socket.
    async fn send_to_engine(
        &self,
        engine_id: &EngineId,
        request_type: Bytes,
        payload: Vec<u8>,
        aux_frames: Vec<Bytes>,
    ) -> Result<()> {
        let mut input_send = self.input_send.lock().await;
        send_message(
            &mut input_send,
            engine_id,
            request_type,
            payload.into(),
            aux_frames,
        )
        .await
    }

    /// Send an Abort for one request id to its engine.
    async fn abort(&self, engine_id: &EngineId, request_id: &str) -> Result<()> {
        let payload = P::encode_abort(request_id)?;
        self.send_to_engine(engine_id, P::abort_frame(), payload, Vec::new())
            .await
    }
}

/// Direct ZMQ connection to a same-host engine (one or more DP ranks behind one
/// shared transport), generic over the engine's wire protocol `P`.
pub struct Client<P: EngineProtocol> {
    inner: Arc<ClientInner<P>>,
    tasks: Vec<JoinHandle<()>>,
}

impl<P: EngineProtocol> Client<P> {
    /// Build a client over a connected transport, spawning the output loop and
    /// dispatcher. Background tasks are aborted when the client is dropped.
    pub fn new(transport: ConnectedTransport) -> Self {
        let ConnectedTransport {
            engines,
            input_send,
            output_socket,
            ..
        } = transport;

        let (abort_tx, abort_rx) = mpsc::unbounded_channel();
        let inner = Arc::new(ClientInner {
            input_send: tokio::sync::Mutex::new(input_send),
            engines,
            registry: Mutex::new(RequestRegistry::default()),
            load: Mutex::new(HashMap::new()),
            abort_tx,
        });

        // Transport output loop: decode raw frames -> EngineBatch channel.
        let (out_tx, out_rx) = mpsc::channel(256);
        #[expect(
            clippy::disallowed_methods,
            reason = "background tasks are aborted on client drop"
        )]
        let output_task = tokio::spawn(run_output_loop::<P>(output_socket, out_tx));
        #[expect(
            clippy::disallowed_methods,
            reason = "background tasks are aborted on client drop"
        )]
        let dispatch_task = tokio::spawn(run_dispatcher::<P>(out_rx, abort_rx, inner.clone()));

        Self {
            inner,
            tasks: vec![output_task, dispatch_task],
        }
    }

    /// The engines connected on this transport.
    pub fn engines(&self) -> &[ConnectedEngine] {
        &self.inner.engines
    }

    /// Whether the connection is still live. Becomes false once the dispatcher
    /// observes `ENGINE_CORE_DEAD`, a transport failure, or the output stream
    /// closing. No RPC — this is a local liveness flag.
    pub fn is_alive(&self) -> bool {
        !self.inner.registry.lock().closed
    }

    /// The latest per-rank load for one engine index (DP routing signal), if
    /// any batch has been seen from it yet.
    pub fn engine_load(&self, engine_index: u32) -> Option<EngineLoad> {
        self.inner.load.lock().get(&engine_index).copied()
    }

    /// Submit a request and return a stream of its outputs. The request is
    /// routed by `data_parallel_rank` (SMG-pinned) or to the sole engine.
    /// Dropping the returned stream before it finishes aborts the request.
    pub async fn submit(&self, request: P::Request) -> Result<RequestStream<P>> {
        P::validate(&request)?;
        let engine_id = self.inner.select_engine(P::data_parallel_rank(&request))?;

        let request_id = P::request_id(&request).to_string();
        let receiver = self.inner.registry.lock().register(request_id.clone())?;

        let (payload, aux_frames) = P::encode_add(&request)?;
        if let Err(error) = self
            .inner
            .send_to_engine(&engine_id, P::add_frame(), payload, aux_frames)
            .await
        {
            // Roll back the registry entry so a failed send doesn't leak it.
            self.inner.registry.lock().remove_all([&request_id]);
            return Err(error);
        }

        Ok(RequestStream {
            request_id,
            engine_id,
            receiver,
            abort_tx: self.inner.abort_tx.clone(),
            finished: false,
        })
    }

    /// Explicitly abort an in-flight request.
    pub async fn abort(&self, engine_id: &EngineId, request_id: &str) -> Result<()> {
        self.inner
            .registry
            .lock()
            .remove_all([&request_id.to_string()]);
        self.inner.abort(engine_id, request_id).await
    }
}

impl<P: EngineProtocol> Drop for Client<P> {
    fn drop(&mut self) {
        for task in &self.tasks {
            task.abort();
        }
    }
}

/// Route decoded outputs to per-request streams and forward auto-abort requests
/// to their engines. Runs until the output channel closes or the engine dies.
/// If the engine emits no output for this long while requests are in flight, the
/// dispatcher treats it as dead. A hard engine death (SIGKILL/OOM) sends no
/// `ENGINE_CORE_DEAD` sentinel, and a bound PULL socket never errors when its
/// PUSH peer vanishes — so silence is the only available death signal. Generous
/// so a slow first token never trips it, while still bounding an otherwise
/// infinite hang.
const ENGINE_SILENCE_DEATH_TIMEOUT: Duration = Duration::from_secs(300);

async fn run_dispatcher<P: EngineProtocol>(
    mut out_rx: mpsc::Receiver<Result<EngineBatch<P::Output>>>,
    mut abort_rx: mpsc::UnboundedReceiver<(EngineId, String)>,
    inner: Arc<ClientInner<P>>,
) {
    loop {
        tokio::select! {
            output = out_rx.recv() => {
                match output {
                    Some(Ok(batch)) => {
                        if let Some(load) = batch.load {
                            inner.load.lock().insert(batch.engine_index, load);
                        }
                        let mut registry = inner.registry.lock();
                        for output in batch.outputs {
                            registry.route(output);
                        }
                        registry.remove_all(&batch.finished_request_ids);
                    }
                    Some(Err(error)) => {
                        if matches!(error, Error::EngineCoreDead | Error::Transport(_)) {
                            warn!(%error, "engine transport failed; failing all in-flight requests");
                            inner.registry.lock().fail_all(Arc::new(error));
                            return;
                        }
                        // A per-message decode error is non-fatal; keep going.
                        warn!(%error, "ignoring undecodable engine output");
                    }
                    None => break,
                }
            }
            abort = abort_rx.recv() => {
                if let Some((engine_id, request_id)) = abort {
                    inner.registry.lock().remove_all([&request_id]);
                    if let Err(error) = inner.abort(&engine_id, &request_id).await {
                        warn!(%error, %request_id, "failed to send abort");
                    }
                }
            }
            // Positive-liveness watchdog. select! rebuilds this sleep every
            // iteration, so any output or abort resets it; it only fires after a
            // full window of pure silence.
            () = tokio::time::sleep(ENGINE_SILENCE_DEATH_TIMEOUT) => {
                let mut registry = inner.registry.lock();
                if !registry.requests.is_empty() {
                    warn!(
                        "no engine output for {ENGINE_SILENCE_DEATH_TIMEOUT:?} with in-flight \
                         requests; treating engine as dead"
                    );
                    registry.fail_all(Arc::new(Error::EngineCoreDead));
                    return;
                }
                // Idle with nothing in flight: silence is expected, keep waiting.
            }
        }
    }
    inner
        .registry
        .lock()
        .fail_all(Arc::new(Error::ClientClosed {
            message: "engine output stream ended".to_string(),
        }));
}

/// Stream of outputs for one submitted request. Dropping it before the terminal
/// output aborts the request on the engine.
pub struct RequestStream<P: EngineProtocol> {
    request_id: String,
    engine_id: EngineId,
    receiver: OutputReceiver<P::Output>,
    abort_tx: mpsc::UnboundedSender<(EngineId, String)>,
    finished: bool,
}

impl<P: EngineProtocol> RequestStream<P> {
    /// The request id this stream tracks.
    pub fn request_id(&self) -> &str {
        &self.request_id
    }
}

impl<P: EngineProtocol> Stream for RequestStream<P> {
    type Item = Result<P::Output>;

    fn poll_next(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Option<Self::Item>> {
        use std::task::Poll;
        if self.finished {
            return Poll::Ready(None);
        }
        match self.receiver.poll_recv(cx) {
            Poll::Ready(Some(Ok(output))) => {
                if output.finished() {
                    self.finished = true;
                }
                Poll::Ready(Some(Ok(output)))
            }
            Poll::Ready(Some(Err(error))) => {
                self.finished = true;
                Poll::Ready(Some(Err(error)))
            }
            Poll::Ready(None) => {
                self.finished = true;
                Poll::Ready(None)
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl<P: EngineProtocol> Drop for RequestStream<P> {
    fn drop(&mut self) {
        if !self.finished {
            // Best-effort auto-abort; the dispatcher sends the Abort frame.
            let _ = self
                .abort_tx
                .send((self.engine_id.clone(), self.request_id.clone()));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use futures::StreamExt;

    use super::*;
    use crate::{
        codec::{decode_msgpack, encode_msgpack},
        mock_engine::{connect_to_frontend, default_ready_response, IpcNamespace, MockEngine},
        protocol::vllm::{
            output::{
                EngineCoreFinishReason, EngineCoreOutput, EngineCoreOutputs, RequestBatchOutputs,
            },
            request::EngineCoreRequest,
            stats::SchedulerStats,
        },
        transport::{connect_handshake, ENGINE_CORE_DEAD_SENTINEL},
    };

    const TIMEOUT: Duration = Duration::from_secs(10);

    /// The returned [`IpcNamespace`] owns the socket tempdir; hold it for the
    /// test duration so the ipc files outlive the client and are cleaned up.
    async fn connect() -> (EngineCoreClient, MockEngine, IpcNamespace) {
        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );
        let (transport, engine) = tokio::join!(
            connect_handshake(
                &handshake,
                1,
                "127.0.0.1",
                Some(&input),
                Some(&output),
                TIMEOUT
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        (
            EngineCoreClient::new(transport.unwrap()),
            engine.unwrap(),
            ns,
        )
    }

    fn batch(engine_index: u32, output: EngineCoreOutput) -> Vec<Bytes> {
        let finished = output.finish_reason.map(|_| {
            let mut set = std::collections::BTreeSet::new();
            set.insert(output.request_id.clone());
            set
        });
        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index,
            outputs: vec![output],
            finished_requests: finished,
            ..RequestBatchOutputs::default()
        });
        vec![Bytes::from(encode_msgpack(&outputs).unwrap())]
    }

    #[tokio::test]
    async fn submit_streams_tokens_until_finish() {
        let (client, mut engine, _ns) = connect().await;

        let request = EngineCoreRequest {
            request_id: "req-1".to_string(),
            prompt_token_ids: Some(vec![1, 2, 3]),
            ..EngineCoreRequest::default()
        };
        let mut stream = client.submit(request).await.unwrap();

        // Engine receives the Add and streams two chunks then a terminal output.
        let frames = engine.recv_request().await.unwrap();
        assert_eq!(frames[0].as_ref(), b"\x00");
        let received: EngineCoreRequest = decode_msgpack(frames[1].as_ref()).unwrap();
        assert_eq!(received.request_id, "req-1");

        engine
            .send_output(batch(
                0,
                EngineCoreOutput {
                    request_id: "req-1".into(),
                    new_token_ids: vec![10],
                    ..Default::default()
                },
            ))
            .await
            .unwrap();
        engine
            .send_output(batch(
                0,
                EngineCoreOutput {
                    request_id: "req-1".into(),
                    new_token_ids: vec![11],
                    finish_reason: Some(EngineCoreFinishReason::Stop),
                    ..Default::default()
                },
            ))
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.new_token_ids, vec![10]);
        assert!(!first.finished());
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.new_token_ids, vec![11]);
        assert!(second.finished());
        // Terminal output ends the stream.
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn scheduler_stats_surface_as_load_signal() {
        let (client, mut engine, _ns) = connect().await;
        let mut stream = client
            .submit(EngineCoreRequest {
                request_id: "r".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        engine.recv_request().await.unwrap();

        let outputs = EngineCoreOutputs::RequestBatch(RequestBatchOutputs {
            engine_index: 0,
            outputs: vec![EngineCoreOutput {
                request_id: "r".into(),
                new_token_ids: vec![7],
                finish_reason: Some(EngineCoreFinishReason::Stop),
                ..Default::default()
            }],
            scheduler_stats: Some(Box::new(SchedulerStats {
                num_running_reqs: 3,
                num_waiting_reqs: 5,
                ..Default::default()
            })),
            finished_requests: Some(std::collections::BTreeSet::from(["r".to_string()])),
            ..Default::default()
        });
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&outputs).unwrap())])
            .await
            .unwrap();

        // Drain the stream so the batch is processed.
        while stream.next().await.is_some() {}
        let load = client.engine_load(0).expect("load recorded");
        assert_eq!(load.num_running, 3);
        assert_eq!(load.num_waiting, 5);
    }

    #[tokio::test]
    async fn dropping_stream_sends_abort() {
        let (client, mut engine, _ns) = connect().await;
        let stream = client
            .submit(EngineCoreRequest {
                request_id: "req-abort".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        // Consume the Add frame.
        let add = engine.recv_request().await.unwrap();
        assert_eq!(add[0].as_ref(), b"\x00");

        drop(stream); // not finished -> auto-abort

        let abort = engine.recv_request().await.unwrap();
        assert_eq!(abort[0].as_ref(), b"\x01"); // Abort
        let ids: Vec<String> = decode_msgpack(abort[1].as_ref()).unwrap();
        assert_eq!(ids, vec!["req-abort".to_string()]);
    }

    #[tokio::test]
    async fn engine_dead_fails_inflight_requests() {
        let (client, mut engine, _ns) = connect().await;
        let mut stream = client
            .submit(EngineCoreRequest {
                request_id: "r".into(),
                ..Default::default()
            })
            .await
            .unwrap();
        engine.recv_request().await.unwrap();

        engine
            .send_output(vec![Bytes::from_static(ENGINE_CORE_DEAD_SENTINEL)])
            .await
            .unwrap();

        let result = stream.next().await.expect("an error item");
        assert!(matches!(result, Err(Error::Shared(_))));
    }

    /// The generic connector drives the TokenSpeed protocol over the same shared
    /// transport: it frames a tagged `TokenizedGenerateReqInput` and streams
    /// `BatchTokenIDOutSlim` batches back until a finish reason is seen.
    #[tokio::test]
    async fn tokenspeed_client_submits_and_streams() {
        use crate::protocol::tokenspeed::{
            output::BatchTokenIDOutSlim,
            request::{TokenSpeedRequestType, TokenizedGenerateReqInput},
            sampling::SamplingParams,
        };

        let ns = IpcNamespace::new().unwrap();
        let (handshake, input, output) = (
            ns.handshake_endpoint(),
            ns.input_endpoint(),
            ns.output_endpoint(),
        );
        std::mem::forget(ns);
        let (transport, engine) = tokio::join!(
            connect_handshake(
                &handshake,
                1,
                "127.0.0.1",
                Some(&input),
                Some(&output),
                TIMEOUT
            ),
            connect_to_frontend(
                &handshake,
                EngineId::from_engine_index(0),
                default_ready_response()
            ),
        );
        let client = TokenSpeedClient::new(transport.unwrap());
        let mut engine = engine.unwrap();

        let request = TokenizedGenerateReqInput {
            rid: "ts-1".to_string(),
            input_ids: vec![1, 2, 3],
            sampling_params: SamplingParams {
                max_new_tokens: Some(4),
                ..SamplingParams::default()
            },
            stream: true,
            ..TokenizedGenerateReqInput::default()
        };
        let mut stream = client.submit(request).await.unwrap();

        // Engine sees the Add frame + the positional-array payload.
        let frames = engine.recv_request().await.unwrap();
        assert_eq!(
            TokenSpeedRequestType::from_frame(frames[0].as_ref()),
            Some(TokenSpeedRequestType::Add)
        );
        let received: TokenizedGenerateReqInput = decode_msgpack(frames[1].as_ref()).unwrap();
        assert_eq!(received.rid, "ts-1");
        assert_eq!(received.input_ids, vec![1, 2, 3]);

        let chunk = BatchTokenIDOutSlim {
            rids: vec!["ts-1".into()],
            output_ids: vec![vec![10]],
            finished_reasons: vec![String::new()],
            prompt_tokens: vec![3],
            completion_tokens: vec![1],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
        };
        let done = BatchTokenIDOutSlim {
            rids: vec!["ts-1".into()],
            output_ids: vec![vec![11]],
            finished_reasons: vec!["stop".into()],
            prompt_tokens: vec![3],
            completion_tokens: vec![2],
            cached_tokens: vec![0],
            output_token_logprobs_val: vec![vec![]],
            output_token_logprobs_idx: vec![vec![]],
        };
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&chunk).unwrap())])
            .await
            .unwrap();
        engine
            .send_output(vec![Bytes::from(encode_msgpack(&done).unwrap())])
            .await
            .unwrap();

        let first = stream.next().await.unwrap().unwrap();
        assert_eq!(first.output_ids, vec![10]);
        assert!(!first.finished());
        let second = stream.next().await.unwrap().unwrap();
        assert_eq!(second.output_ids, vec![11]);
        assert_eq!(second.finish_reason.as_deref(), Some("stop"));
        assert!(second.finished());
        // Terminal output ends the stream.
        assert!(stream.next().await.is_none());
    }
}
