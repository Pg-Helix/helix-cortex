// Package api is Workflow 2's (ADR-0007) HTTP surface: the handlers
// behind authMiddleware, implementing the ingest/document/permission/
// evidence/search/answer operations from helix-protocol's OpenAPI
// contract. Deliberately not the entire contract: listModels/getModel
// and the standalone ingest-status endpoint are out of scope for this
// milestone (not needed for Workflow 2's user-visible outcome) and have
// no handler here.
package api

import (
	"encoding/json"
	"errors"
	"io"
	"net/http"

	"github.com/Pg-Helix/helix-cortex/internal/auth"
	"github.com/Pg-Helix/helix-cortex/internal/store"
	"github.com/Pg-Helix/helix-cortex/internal/testhooks"
	"github.com/Pg-Helix/helix-infer/embed"
	protocol "github.com/Pg-Helix/helix-protocol/gen/go"
)

// Server holds everything a handler needs. TestHooks is nil unless the
// server was started with CORTEX_TEST_HOOKS=1 (cmd/cortex-api) -- a nil
// Registry means barrierPause is never invoked, so the pause-for-
// in-flight-revocation mechanism is fully inert in a normal deployment.
type Server struct {
	Store         *store.Store
	Authenticator auth.Authenticator
	Embedder      *embed.Guarded
	TargetModel   string
	TestHooks     *testhooks.Registry
}

func (s *Server) Handler() http.Handler {
	business := http.NewServeMux()
	business.HandleFunc("POST /v1/collections", s.createCollection)
	business.HandleFunc("POST /v1/collections/{collectionId}/documents", s.ingestDocument)
	business.HandleFunc("GET /v1/collections/{collectionId}/documents/{documentId}", s.getDocument)
	business.HandleFunc("DELETE /v1/collections/{collectionId}/documents/{documentId}", s.deleteDocument)
	business.HandleFunc("GET /v1/collections/{collectionId}/permissions", s.listPermissions)
	business.HandleFunc("POST /v1/collections/{collectionId}/permissions", s.grantPermission)
	business.HandleFunc("DELETE /v1/collections/{collectionId}/permissions/{principal}", s.revokePermission)
	business.HandleFunc("GET /v1/evidence/{chunkId}", s.getEvidence)
	business.HandleFunc("POST /v1/search", s.search)
	business.HandleFunc("POST /v1/answers", s.getAnswer)

	top := http.NewServeMux()
	top.Handle("/v1/", authMiddleware(s.Authenticator, business))
	if s.TestHooks != nil {
		// Deliberately NOT behind authMiddleware: these exist only for
		// the acceptance suite's deterministic in-flight-revocation
		// barrier (ADR-0007) to drive directly, and are only even
		// registered when CORTEX_TEST_HOOKS=1 is explicitly set (see
		// cmd/cortex-api) -- never present in a normal deployment.
		// Registered under /v1/ (matching how the client's base URL
		// actually addresses them) but directly on top, not business:
		// ServeMux gives this exact pattern precedence over the
		// "/v1/" subtree pattern above, so it bypasses authMiddleware
		// for exactly these two routes without needing a path rewrite.
		top.HandleFunc("GET /v1/__test/barrier/{token}/status", s.testBarrierStatus)
		top.HandleFunc("POST /v1/__test/barrier/{token}/release", s.testBarrierRelease)
	}
	return top
}

// checkTenant implements ADR-0007 decision 1: a request body's tenant
// field is data, never authority. The authenticated principal's own
// tenant is the only one ever trusted; a body that disagrees is rejected
// outright rather than silently overridden.
func checkTenant(bodyTenant string, p auth.Principal) error {
	if bodyTenant != "" && bodyTenant != p.Tenant {
		return store.ErrTenantMismatch
	}
	return nil
}

func decodeJSON(r *http.Request, v any) error {
	defer r.Body.Close()
	dec := json.NewDecoder(io.LimitReader(r.Body, 2<<20)) // 2 MiB request body cap
	return dec.Decode(v)
}

func (s *Server) createCollection(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	var body protocol.CreateCollectionJSONBody
	if err := decodeJSON(r, &body); err != nil {
		writeError(w, badRequest("malformed request body"))
		return
	}
	if err := checkTenant(body.Tenant, p); err != nil {
		writeError(w, err)
		return
	}
	if body.Name == "" {
		writeError(w, badRequest("name is required"))
		return
	}
	col, err := s.Store.CreateCollection(r.Context(), p, body.Name)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, col)
}

func (s *Server) ingestDocument(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	collectionID := r.PathValue("collectionId")
	var body protocol.IngestDocumentJSONBody
	if err := decodeJSON(r, &body); err != nil {
		writeError(w, badRequest("malformed request body"))
		return
	}
	if err := body.Validate(); err != nil {
		writeError(w, badRequest(err.Error()))
		return
	}
	if err := checkTenant(body.Tenant, p); err != nil {
		writeError(w, err)
		return
	}
	doc, err := s.Store.IngestDocument(r.Context(), p, collectionID, body.Name, string(body.ContentType), body.Content)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, doc)
}

func (s *Server) getDocument(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	doc, err := s.Store.GetDocument(r.Context(), p, r.PathValue("collectionId"), r.PathValue("documentId"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, doc)
}

func (s *Server) deleteDocument(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	if err := s.Store.DeleteDocument(r.Context(), p, r.PathValue("collectionId"), r.PathValue("documentId")); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) listPermissions(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	perms, err := s.Store.ListPermissions(r.Context(), p, r.PathValue("collectionId"))
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, struct {
		Permissions []protocol.Permission `json:"permissions"`
	}{perms})
}

func (s *Server) grantPermission(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	var body protocol.Permission
	if err := decodeJSON(r, &body); err != nil {
		writeError(w, badRequest("malformed request body"))
		return
	}
	if err := body.Validate(); err != nil {
		writeError(w, badRequest(err.Error()))
		return
	}
	granted, err := s.Store.GrantPermission(r.Context(), p, r.PathValue("collectionId"), body)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusCreated, granted)
}

func (s *Server) revokePermission(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	if err := s.Store.RevokePermission(r.Context(), p, r.PathValue("collectionId"), r.PathValue("principal")); err != nil {
		writeError(w, err)
		return
	}
	w.WriteHeader(http.StatusNoContent)
}

func (s *Server) getEvidence(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	if err := checkTenant(r.URL.Query().Get("tenant"), p); err != nil {
		writeError(w, err)
		return
	}
	ev, err := s.Store.GetEvidence(r.Context(), p, r.PathValue("chunkId"))
	if err != nil {
		if errors.Is(err, store.ErrNotFound) {
			writeError(w, citationInvalid{})
			return
		}
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, ev)
}

func (s *Server) search(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	var body protocol.SearchJSONBody
	if err := decodeJSON(r, &body); err != nil {
		writeError(w, badRequest("malformed request body"))
		return
	}
	if err := checkTenant(body.Tenant, p); err != nil {
		writeError(w, err)
		return
	}
	topK := 10
	if body.TopK != nil {
		topK = *body.TopK
	}
	var results []protocol.SearchResult
	var err error
	// X-Eval-Stage: test-only instrumentation for helix-labs/eval, same
	// gate as the in-flight-revocation barrier -- never reachable unless
	// CORTEX_TEST_HOOKS=1.
	if s.TestHooks != nil && r.Header.Get("X-Eval-Stage") != "" {
		results, err = s.Store.SearchStage(r.Context(), p, body.CollectionId, body.Query, topK, s.Embedder, s.TargetModel, r.Header.Get("X-Eval-Stage"))
	} else {
		results, err = s.Store.Search(r.Context(), p, body.CollectionId, body.Query, topK, s.Embedder, s.TargetModel)
	}
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, struct {
		Results []protocol.SearchResult `json:"results"`
	}{results})
}

func (s *Server) getAnswer(w http.ResponseWriter, r *http.Request) {
	p := principalFromContext(r.Context())
	var body protocol.GetAnswerJSONBody
	if err := decodeJSON(r, &body); err != nil {
		writeError(w, badRequest("malformed request body"))
		return
	}
	if err := checkTenant(body.Tenant, p); err != nil {
		writeError(w, err)
		return
	}

	var barrierPause func()
	if s.TestHooks != nil {
		if token := r.Header.Get("X-Test-Barrier-Token"); token != "" {
			barrierPause = func() { s.TestHooks.Pause(token) }
		}
	}

	ans, err := s.Store.Answer(r.Context(), p, body.CollectionId, body.Query, s.Embedder, s.TargetModel, barrierPause)
	if err != nil {
		writeError(w, err)
		return
	}
	writeJSON(w, http.StatusOK, ans)
}

func (s *Server) testBarrierStatus(w http.ResponseWriter, r *http.Request) {
	token := r.PathValue("token")
	writeJSON(w, http.StatusOK, struct {
		Arrived bool `json:"arrived"`
	}{s.TestHooks.Arrived(token)})
}

func (s *Server) testBarrierRelease(w http.ResponseWriter, r *http.Request) {
	s.TestHooks.Release(r.PathValue("token"))
	w.WriteHeader(http.StatusNoContent)
}

type badRequestError string

func (e badRequestError) Error() string { return string(e) }
func badRequest(msg string) error       { return badRequestError(msg) }

type citationInvalid struct{}

func (citationInvalid) Error() string { return "citation_invalid" }
