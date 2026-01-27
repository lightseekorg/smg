package service

import (
	smg "github.com/lightseek/smg/go-grpc-sdk"
)

// SMGService wraps SMG client
type SMGService struct {
	client *smg.Client
}

func NewSMGService(endpoint, tokenizerPath string) (*SMGService, error) {
	client, err := smg.NewClient(smg.ClientConfig{
		Endpoint:      endpoint,
		TokenizerPath: tokenizerPath,
	})
	if err != nil {
		return nil, err
	}

	return &SMGService{
		client: client,
	}, nil
}

// Client returns the underlying SMG client
func (s *SMGService) Client() *smg.Client {
	return s.client
}

// Close closes the SMG client
func (s *SMGService) Close() error {
	if s.client != nil {
		return s.client.Close()
	}
	return nil
}
