//! Mock gRPC worker implementing the TokenSpeed scheduler service. The gateway
//! tokenizes and sends token ids; this service streams back canned token ids.

use std::{
    net::{IpAddr, SocketAddr},
    pin::Pin,
    sync::Arc,
};

use futures::{stream, Stream};
use smg_grpc_client::{common_proto as common, tokenspeed_scheduler::tokenspeed_proto as ts};
use tonic::{transport::Server, Request, Response, Status};
use ts::{
    generate_response::Response as GenResp,
    token_speed_scheduler_server::{TokenSpeedScheduler, TokenSpeedSchedulerServer},
};

use crate::config::Config;

/// Serve the mock TokenSpeed gRPC service on `port` until the process exits.
pub async fn serve(cfg: Arc<Config>, host: String, port: u16) {
    let ip = match host.parse::<IpAddr>() {
        Ok(ip) => ip,
        Err(e) => {
            tracing::error!("grpc worker host {host} invalid: {e}");
            return;
        }
    };
    let addr = SocketAddr::new(ip, port);
    let service = MockScheduler { cfg };
    if let Err(e) = Server::builder()
        .add_service(TokenSpeedSchedulerServer::new(service))
        .serve(addr)
        .await
    {
        tracing::error!("grpc worker {port} stopped: {e}");
    }
}

#[derive(Clone)]
struct MockScheduler {
    cfg: Arc<Config>,
}

type GenStream = Pin<Box<dyn Stream<Item = Result<ts::GenerateResponse, Status>> + Send>>;

#[tonic::async_trait]
impl TokenSpeedScheduler for MockScheduler {
    type GenerateStream = GenStream;

    async fn generate(
        &self,
        request: Request<ts::GenerateRequest>,
    ) -> Result<Response<Self::GenerateStream>, Status> {
        let request_id = request.into_inner().request_id;
        if !self.cfg.gen_delay.is_zero() {
            tokio::time::sleep(self.cfg.gen_delay).await;
        }
        let ids: Vec<u32> = (0..self.cfg.output_tokens).map(|i| 100 + i).collect();

        let mut items: Vec<Result<ts::GenerateResponse, Status>> = Vec::new();
        for id in &ids {
            items.push(Ok(ts::GenerateResponse {
                request_id: request_id.clone(),
                response: Some(GenResp::Chunk(ts::GenerateStreamChunk {
                    token_ids: vec![*id],
                    prompt_tokens: 1,
                    completion_tokens: 1,
                    cached_tokens: 0,
                    output_logprobs: None,
                    index: 0,
                })),
            }));
        }
        items.push(Ok(ts::GenerateResponse {
            request_id,
            response: Some(GenResp::Complete(ts::GenerateComplete {
                output_ids: ids,
                finish_reason: "stop".to_string(),
                prompt_tokens: 1,
                completion_tokens: self.cfg.output_tokens,
                cached_tokens: 0,
                output_logprobs: None,
                matched_stop: None,
                index: 0,
            })),
        }));

        Ok(Response::new(Box::pin(stream::iter(items))))
    }

    async fn health_check(
        &self,
        _request: Request<ts::HealthCheckRequest>,
    ) -> Result<Response<ts::HealthCheckResponse>, Status> {
        Ok(Response::new(ts::HealthCheckResponse {
            healthy: true,
            message: "ok".to_string(),
        }))
    }

    async fn abort(
        &self,
        _request: Request<ts::AbortRequest>,
    ) -> Result<Response<ts::AbortResponse>, Status> {
        Ok(Response::new(ts::AbortResponse {
            success: true,
            message: String::new(),
        }))
    }

    async fn get_model_info(
        &self,
        _request: Request<ts::GetModelInfoRequest>,
    ) -> Result<Response<ts::GetModelInfoResponse>, Status> {
        Ok(Response::new(ts::GetModelInfoResponse {
            model_path: self.cfg.model_id.clone(),
            tokenizer_path: self.cfg.tokenizer_path.clone(),
            served_model_name: self.cfg.model_id.clone(),
            model_type: "mock".to_string(),
            architectures: vec!["MockForCausalLM".to_string()],
            max_context_length: 32768,
            max_req_input_len: 32768,
            vocab_size: 32000,
            eos_token_ids: vec![2],
            pad_token_id: 0,
            bos_token_id: 1,
            weight_version: "mock".to_string(),
            default_sampling_params_json: String::new(),
            supports_vision: false,
        }))
    }

    async fn get_server_info(
        &self,
        _request: Request<ts::GetServerInfoRequest>,
    ) -> Result<Response<ts::GetServerInfoResponse>, Status> {
        Ok(Response::new(ts::GetServerInfoResponse {
            server_args: None,
            scheduler_info: None,
            active_requests: 0,
            is_paused: false,
            uptime_seconds: 0.0,
            max_total_num_tokens: 1_000_000,
            tokenspeed_version: "mock".to_string(),
            start_time: None,
        }))
    }

    async fn get_loads(
        &self,
        _request: Request<ts::GetLoadsRequest>,
    ) -> Result<Response<ts::GetLoadsResponse>, Status> {
        Ok(Response::new(ts::GetLoadsResponse {
            timestamp: String::new(),
            version: "mock".to_string(),
            dp_rank_count: 1,
            loads: vec![ts::SchedulerLoad {
                dp_rank: 0,
                num_running_reqs: 0,
                num_waiting_reqs: 0,
                num_total_reqs: 0,
                num_used_tokens: 0,
                max_total_num_tokens: 1_000_000,
                max_running_requests: 0,
                token_usage: 0.0,
                gen_throughput: 0.0,
                cache_hit_rate: 0.0,
                utilization: 0.0,
                memory: None,
                queues: None,
            }],
            aggregate: None,
        }))
    }

    async fn flush_cache(
        &self,
        _request: Request<common::FlushCacheRequest>,
    ) -> Result<Response<common::FlushCacheResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn start_profile(
        &self,
        _request: Request<common::StartProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }

    async fn stop_profile(
        &self,
        _request: Request<common::StopProfileRequest>,
    ) -> Result<Response<common::ProfileResponse>, Status> {
        Err(Status::unimplemented("mock-worker"))
    }
}
