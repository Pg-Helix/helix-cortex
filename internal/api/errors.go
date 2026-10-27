package api

import (
	"encoding/json"
	"errors"
	"net/http"

	"github.com/Pg-Helix/helix-cortex/internal/ingest"
	"github.com/Pg-Helix/helix-cortex/internal/store"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
	"github.com/google/uuid"
)

// writeError maps an internal error to the ErrorResponse shape every
// non-2xx response uses (errors/codes.md), choosing an HTTP status and
// stable code -- never leaking which specific case applied beyond what
// the code itself already implies (e.g. citation_invalid covers
// revoked/superseded/deleted/fabricated alike).
func writeError(w http.ResponseWriter, err error) {
	code, status, message := classifyError(err)
	writeJSON(w, status, protocol.ErrorResponse{
		Code:    code,
		Message: message,
		TraceId: uuid.NewString(),
	})
}

func classifyError(err error) (code string, status int, message string) {
	var tooLarge *ingest.ErrTooLarge
	var badReq badRequestError
	switch {
	case errors.Is(err, authError{}):
		return "unauthorized", http.StatusUnauthorized, "missing or invalid bearer token"
	case errors.As(err, &badReq):
		return "invalid_argument", http.StatusBadRequest, badReq.Error()
	case errors.Is(err, citationInvalid{}):
		return "citation_invalid", http.StatusNotFound, "this citation no longer matches current state"
	case errors.Is(err, store.ErrForbidden):
		return "forbidden", http.StatusForbidden, "the authenticated principal does not hold the required grant"
	case errors.Is(err, store.ErrNotFound):
		return "not_found", http.StatusNotFound, "the requested resource does not exist or is not visible to the caller"
	case errors.Is(err, store.ErrTenantMismatch):
		return "tenant_mismatch", http.StatusForbidden, "request tenant does not match the authenticated principal's tenant"
	case errors.Is(err, store.ErrUnsupportedContentType):
		return "unsupported_content_type", http.StatusBadRequest, "content_type must be text/plain or text/markdown"
	case errors.As(err, &tooLarge):
		// Both content_too_large and chunk_limit_exceeded (errors/codes.md)
		// come through ingest.ErrTooLarge; the message distinguishes them
		// for a human, the code intentionally does not -- both mean "no,
		// and don't retry without changing the input."
		return "content_too_large", http.StatusBadRequest, tooLarge.Error()
	case err == nil:
		return "internal", http.StatusInternalServerError, "unexpected nil error"
	default:
		return "internal", http.StatusInternalServerError, "internal error"
	}
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}
