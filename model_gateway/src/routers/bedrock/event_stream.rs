//! Minimal decoder for AWS binary event streams (`application/vnd.amazon.eventstream`)
//! used by Bedrock `ConverseStream`.

use bytes::{Buf, BytesMut};

/// One decoded Bedrock stream event (payload is the raw JSON union body).
#[derive(Debug, Clone)]
pub(crate) struct StreamEvent {
    pub event_type: String,
    pub payload: Vec<u8>,
}

#[derive(Debug)]
pub(crate) enum DecodeError {
    Truncated,
    Invalid(&'static str),
}

/// Try to pull the next complete message from `buf`. On success, advances `buf`.
pub(crate) fn pop_next_event(buf: &mut BytesMut) -> Result<StreamEvent, DecodeError> {
    if buf.len() < 12 {
        return Err(DecodeError::Truncated);
    }

    let total_len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let headers_len = u32::from_be_bytes([buf[4], buf[5], buf[6], buf[7]]) as usize;
    let prelude_crc = u32::from_be_bytes([buf[8], buf[9], buf[10], buf[11]]);

    if total_len < 16 {
        return Err(DecodeError::Invalid("message too short"));
    }
    if buf.len() < total_len {
        return Err(DecodeError::Truncated);
    }

    let prelude_crc_calc = crc32fast::hash(&buf[0..8]);
    if prelude_crc_calc != prelude_crc {
        return Err(DecodeError::Invalid("prelude crc mismatch"));
    }

    let headers_end = 12 + headers_len;
    if headers_end + 4 > total_len {
        return Err(DecodeError::Invalid("header length inconsistent"));
    }

    let payload_end = total_len - 4;
    let msg_crc_expected = u32::from_be_bytes([
        buf[payload_end],
        buf[payload_end + 1],
        buf[payload_end + 2],
        buf[payload_end + 3],
    ]);
    let msg_crc_calc = crc32fast::hash(&buf[0..payload_end]);
    if msg_crc_calc != msg_crc_expected {
        return Err(DecodeError::Invalid("message crc mismatch"));
    }

    let headers_bytes = &buf[12..headers_end];
    let payload = buf[headers_end..payload_end].to_vec();

    let event_type = parse_event_type_header(headers_bytes).unwrap_or_default();

    buf.advance(total_len);

    Ok(StreamEvent {
        event_type,
        payload,
    })
}

fn parse_event_type_header(headers: &[u8]) -> Option<String> {
    let mut pos = 0;
    let mut event_type: Option<String> = None;
    while pos < headers.len() {
        let name_len = *headers.get(pos)? as usize;
        pos += 1;
        let name = headers.get(pos..pos + name_len)?;
        pos += name_len;
        let value_type = *headers.get(pos)?;
        pos += 1;
        let vlen = header_value_byte_len(value_type, headers.get(pos..)?)?;
        let value_bytes = headers.get(pos..pos + vlen)?;
        pos += vlen;
        if name == b":event-type" || name == b"event-type" {
            if value_type == 7 && value_bytes.len() >= 2 {
                let slen = u16::from_be_bytes([value_bytes[0], value_bytes[1]]) as usize;
                if value_bytes.len() >= 2 + slen {
                    let s = value_bytes.get(2..2 + slen)?;
                    event_type = std::str::from_utf8(s).ok().map(str::to_owned);
                }
            } else {
                event_type = Some(String::from_utf8_lossy(value_bytes).into_owned());
            }
        }
    }
    event_type
}

/// Byte length of the header **value** for AWS event-stream header value types.
fn header_value_byte_len(value_type: u8, rest: &[u8]) -> Option<usize> {
    match value_type {
        0 | 1 => Some(0),
        2 => Some(1),
        3 => Some(2),
        4 => Some(4),
        5 => Some(8),
        6 => {
            let len = u16::from_be_bytes([*rest.first()?, *rest.get(1)?]) as usize;
            Some(2 + len)
        }
        7 => {
            let len = u16::from_be_bytes([*rest.first()?, *rest.get(1)?]) as usize;
            Some(2 + len)
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_single_event_with_string_header_and_json_payload() {
        let name = b":event-type";
        assert_eq!(name.len(), 11);
        let value_type = 7u8;
        let ev = b"contentBlockDelta";
        let mut value_bytes = Vec::new();
        value_bytes.extend_from_slice(&(ev.len() as u16).to_be_bytes());
        value_bytes.extend_from_slice(ev);

        let mut headers = Vec::new();
        headers.push(name.len() as u8);
        headers.extend_from_slice(name);
        headers.push(value_type);
        headers.extend_from_slice(&value_bytes);

        let headers_len = headers.len();
        let payload = br#"{"contentBlockDelta":{"contentBlockIndex":0,"delta":{"text":"Hi"}}}"#;
        let total_len = 12 + headers_len + payload.len() + 4;
        let mut msg = Vec::new();
        msg.extend_from_slice(&(total_len as u32).to_be_bytes());
        msg.extend_from_slice(&(headers_len as u32).to_be_bytes());
        let prelude_crc = crc32fast::hash(&msg[0..8]);
        msg.extend_from_slice(&prelude_crc.to_be_bytes());
        msg.extend_from_slice(&headers);
        msg.extend_from_slice(payload);
        let msg_crc = crc32fast::hash(&msg[..msg.len()]);
        msg.extend_from_slice(&msg_crc.to_be_bytes());

        let mut buf = BytesMut::from(&msg[..]);
        let ev = pop_next_event(&mut buf).expect("decode");
        assert_eq!(ev.event_type, "contentBlockDelta");
        assert!(buf.is_empty());
    }
}
