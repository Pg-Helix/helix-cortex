package api

import (
	"context"
	"net/http"
	"strings"

	"github.com/Pg-Helix/helix-cortex/internal/auth"
)

type principalContextKey struct{}

// authMiddleware is the real authentication boundary (ADR-0007 decision
// 1): every request must carry a Bearer token the configured
// Authenticator recognizes. There is no bypass -- a nil/misconfigured
// Authenticator was already refused at startup (see cmd/cortex-api), so
// reaching this middleware at all implies auth is actually configured.
func authMiddleware(authenticator auth.Authenticator, next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		header := r.Header.Get("Authorization")
		token := ""
		if strings.HasPrefix(header, "Bearer ") {
			token = strings.TrimPrefix(header, "Bearer ")
		}
		p, err := authenticator.Authenticate(r.Context(), token)
		if err != nil {
			writeError(w, authError{})
			return
		}
		ctx := context.WithValue(r.Context(), principalContextKey{}, p)
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

// authError is a distinct type (not auth.ErrInvalidToken directly) so
// writeError's classifier can special-case it without importing auth
// into errors.go's error-classification switch for just one case.
type authError struct{}

func (authError) Error() string { return "unauthorized" }

func principalFromContext(ctx context.Context) auth.Principal {
	p, _ := ctx.Value(principalContextKey{}).(auth.Principal)
	return p
}
