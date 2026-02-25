use smg_client::SmgError;

#[test]
fn test_error_from_status_400() {
    let body = r#"{"error":{"message":"Invalid model","type":"invalid_request_error","param":"model","code":null}}"#;
    let err = SmgError::from_status(400, body);
    match err {
        SmgError::BadRequest {
            message,
            status,
            body,
        } => {
            assert_eq!(status, 400);
            assert_eq!(message, "Invalid model");
            assert!(body.is_some());
        }
        _ => panic!("expected BadRequest, got {err:?}"),
    }
}

#[test]
fn test_error_from_status_401() {
    let err = SmgError::from_status(401, "Unauthorized");
    match err {
        SmgError::Authentication { status, .. } => assert_eq!(status, 401),
        _ => panic!("expected Authentication, got {err:?}"),
    }
}

#[test]
fn test_error_from_status_429() {
    let err = SmgError::from_status(429, "Rate limited");
    match err {
        SmgError::RateLimit { status, .. } => assert_eq!(status, 429),
        _ => panic!("expected RateLimit, got {err:?}"),
    }
}

#[test]
fn test_error_from_status_500() {
    let err = SmgError::from_status(500, "Internal server error");
    match err {
        SmgError::Server { status, .. } => assert_eq!(status, 500),
        _ => panic!("expected Server, got {err:?}"),
    }
}

#[test]
fn test_error_from_status_plain_text() {
    let err = SmgError::from_status(400, "not json");
    match err {
        SmgError::BadRequest { message, body, .. } => {
            assert_eq!(message, "not json");
            assert!(body.is_none());
        }
        _ => panic!("expected BadRequest, got {err:?}"),
    }
}
