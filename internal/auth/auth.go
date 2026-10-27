// Package auth is the Workflow 2 (ADR-0007) authentication boundary. An
// Authenticator maps an opaque bearer token to exactly one authenticated
// Principal; the server never trusts a client-supplied tenant or
// principal value as authority -- only what an Authenticator returns.
package auth

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
)

// Principal is the authenticated caller: a stable identity within
// exactly one tenant. Request bodies may still carry a tenant field
// (matching existing protocol schemas), but the server rejects it with
// tenant_mismatch if it disagrees with Principal.Tenant -- a body field
// is data, never authority (ADR-0007 decision 1).
type Principal struct {
	Name   string
	Tenant string
}

// Authenticator resolves an opaque bearer token to a Principal.
// ErrInvalidToken (including an empty token) must be returned rather
// than a zero-value Principal -- callers must fail closed, not treat an
// unrecognized token as "anonymous."
type Authenticator interface {
	Authenticate(ctx context.Context, token string) (Principal, error)
}

// ErrInvalidToken is returned for a missing, empty, or unrecognized
// token, or when the Authenticator itself is unconfigured. There is no
// "no auth configured" open mode: an empty or missing fixture rejects
// every request.
var ErrInvalidToken = fmt.Errorf("auth: invalid or unrecognized bearer token")

// LocalFixtureAuthenticator is the local-only, generated, opaque-token
// fixture (ADR-0007 decision 1) -- never a real identity provider. Gated
// by the caller requiring CORTEX_AUTH_PROFILE=local-fixture before this
// type is even constructed (see cmd/cortex-api/main.go); this type itself
// additionally fails closed if its token map ends up empty, so a missing
// or empty fixture file can never silently become "no auth required."
type LocalFixtureAuthenticator struct {
	tokens map[string]Principal
}

// NewLocalFixtureAuthenticator loads token->Principal mappings from a
// JSON file shaped {"<token>": {"principal": "...", "tenant": "..."}}.
// Returns an error if the file is missing, malformed, or maps to zero
// principals -- an empty/absent fixture is a configuration error, not an
// open-access default.
func NewLocalFixtureAuthenticator(path string) (*LocalFixtureAuthenticator, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("auth: read fixture %s: %w", path, err)
	}
	var entries map[string]struct {
		Principal string `json:"principal"`
		Tenant    string `json:"tenant"`
	}
	if err := json.Unmarshal(raw, &entries); err != nil {
		return nil, fmt.Errorf("auth: parse fixture %s: %w", path, err)
	}
	if len(entries) == 0 {
		return nil, fmt.Errorf("auth: fixture %s defines zero principals -- refusing to start with no configured auth", path)
	}
	tokens := make(map[string]Principal, len(entries))
	for token, e := range entries {
		if token == "" || e.Principal == "" || e.Tenant == "" {
			return nil, fmt.Errorf("auth: fixture %s has an entry with an empty token, principal, or tenant", path)
		}
		tokens[token] = Principal{Name: e.Principal, Tenant: e.Tenant}
	}
	return &LocalFixtureAuthenticator{tokens: tokens}, nil
}

func (a *LocalFixtureAuthenticator) Authenticate(_ context.Context, token string) (Principal, error) {
	if token == "" {
		return Principal{}, ErrInvalidToken
	}
	p, ok := a.tokens[token]
	if !ok {
		return Principal{}, ErrInvalidToken
	}
	return p, nil
}
