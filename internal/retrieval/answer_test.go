package retrieval

import "testing"

func TestAssembleExtractiveAnswer_JoinsUpToLimit(t *testing.T) {
	passages := []Passage{
		{ChunkID: "a", Text: "first passage"},
		{ChunkID: "b", Text: "second passage"},
		{ChunkID: "c", Text: "third passage"},
		{ChunkID: "d", Text: "fourth passage, should be excluded by MaxAnswerChunks"},
	}
	answer, used := AssembleExtractiveAnswer(passages)
	want := "first passage ... second passage ... third passage"
	if answer != want {
		t.Fatalf("answer = %q, want %q", answer, want)
	}
	if len(used) != 3 {
		t.Fatalf("used = %d passages, want 3", len(used))
	}
	for i, id := range []string{"a", "b", "c"} {
		if used[i].ChunkID != id {
			t.Fatalf("used[%d].ChunkID = %q, want %q", i, used[i].ChunkID, id)
		}
	}
}

func TestAssembleExtractiveAnswer_TruncatesAtCharLimit(t *testing.T) {
	long := make([]byte, MaxAnswerChars+500)
	for i := range long {
		long[i] = 'x'
	}
	passages := []Passage{{ChunkID: "a", Text: string(long)}}
	answer, used := AssembleExtractiveAnswer(passages)
	if len(answer) != MaxAnswerChars {
		t.Fatalf("answer length = %d, want %d", len(answer), MaxAnswerChars)
	}
	if len(used) != 1 || len(used[0].Text) != MaxAnswerChars {
		t.Fatalf("used[0] text length = %d, want %d", len(used[0].Text), MaxAnswerChars)
	}
}

func TestAssembleExtractiveAnswer_SkipsEmptyPassages(t *testing.T) {
	passages := []Passage{
		{ChunkID: "a", Text: "   "},
		{ChunkID: "b", Text: "real content"},
	}
	answer, used := AssembleExtractiveAnswer(passages)
	if answer != "real content" {
		t.Fatalf("answer = %q, want %q", answer, "real content")
	}
	if len(used) != 1 || used[0].ChunkID != "b" {
		t.Fatalf("used = %+v, want only chunk b", used)
	}
}
