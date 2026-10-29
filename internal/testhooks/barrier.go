// Package testhooks provides a deterministic rendezvous point for the
// in-flight-revocation acceptance scenario (ADR-0007/Workflow 2): "pause
// after candidate selection, commit revoke, resume before final
// authorization/render" -- a real barrier, not timing luck, mirroring
// helix-kernel's own advisory-lock gate pattern at the HTTP-application
// layer instead of inside PostgreSQL.
//
// Entirely in-memory, entirely opt-in: nothing calls Pause unless the
// caller explicitly passes a non-empty token, and the server only wires
// a Registry at all when CORTEX_TEST_HOOKS=1 (see cmd/cortex-api).
package testhooks

import "sync"

type Registry struct {
	mu      sync.Mutex
	arrived map[string]bool
	release map[string]chan struct{}
	closed  map[string]bool
}

func NewRegistry() *Registry {
	return &Registry{
		arrived: make(map[string]bool),
		release: make(map[string]chan struct{}),
		closed:  make(map[string]bool),
	}
}

func (r *Registry) channelFor(token string) chan struct{} {
	r.mu.Lock()
	defer r.mu.Unlock()
	ch, ok := r.release[token]
	if !ok {
		ch = make(chan struct{})
		r.release[token] = ch
	}
	return ch
}

// Pause marks token as arrived and blocks the calling goroutine until
// Release(token) is called. Safe for concurrent use across distinct
// tokens; a given token is meant to be used by exactly one in-flight
// request.
func (r *Registry) Pause(token string) {
	ch := r.channelFor(token)
	r.mu.Lock()
	r.arrived[token] = true
	r.mu.Unlock()
	<-ch
}

// Arrived reports whether token has reached its Pause point yet -- what
// the test harness polls before committing its revoke and calling Release.
func (r *Registry) Arrived(token string) bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.arrived[token]
}

// Release unblocks any goroutine paused on token. Safe to call even if
// nothing has called Pause(token) yet (the paused goroutine will see the
// already-closed channel and proceed immediately) or to call more than
// once for the same token.
func (r *Registry) Release(token string) {
	ch := r.channelFor(token)
	r.mu.Lock()
	defer r.mu.Unlock()
	if !r.closed[token] {
		close(ch)
		r.closed[token] = true
	}
}
