import React, { useState } from "react";
import {
  View,
  Text,
  TextInput,
  Button,
  ScrollView,
  Platform,
  KeyboardAvoidingView,
} from "react-native";

const API_BASE = process.env.EXPO_PUBLIC_API_BASE!;

type RagChunk = {
  id: string;
  text: string;
  source?: string;
  chunk_index?: number;
  distance?: number;
};

type RagQueryResponse = {
  answer: string;
  chunks: RagChunk[];
};

export default function HomeScreen() {
  const [docText, setDocText] = useState("");
  const [docSource, setDocSource] = useState("manual");
  const [ingestStatus, setIngestStatus] = useState<string | null>(null);
  const [ingestLoading, setIngestLoading] = useState(false);

  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [chunks, setChunks] = useState<RagChunk[]>([]);
  const [queryLoading, setQueryLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleIngest = async () => {
    if (!docText.trim()) {
      setIngestStatus("Please enter some text to ingest.");
      return;
    }
    setIngestLoading(true);
    setIngestStatus(null);
    setError(null);

    try {
      const res = await fetch(`${API_BASE}/rag/ingest`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          documents: [docText],
        }),
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Ingest failed: ${res.status} ${body}`);
      }

      const data = await res.json(); // { ingested: number }
      setIngestStatus(`Ingested documents: ${data.ingested}`);
    } catch (e: any) {
      console.error(e);
      setError(e.message ?? String(e));
      setIngestStatus("Ingest failed.");
    } finally {
      setIngestLoading(false);
    }
  };

  const handleQuery = async () => {
    if (!question.trim()) {
      setError("Please enter a question.");
      return;
    }

    setQueryLoading(true);
    setError(null);
    setAnswer("");
    setChunks([]);

    try {
      const res = await fetch(`${API_BASE}/rag/query`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question,
          top_k: 5,
        }),
      });

      if (!res.ok) {
        const body = await res.text();
        throw new Error(`Query failed: ${res.status} ${body}`);
      }

      const data: RagQueryResponse = await res.json();
      setAnswer(data.answer);
      setChunks(data.chunks ?? []);
    } catch (e: any) {
      console.error(e);
      setError(e.message ?? String(e));
    } finally {
      setQueryLoading(false);
    }
  };

  return (
    <KeyboardAvoidingView
      style={{ flex: 1, backgroundColor: "#f5f5f5" }}
      behavior={Platform.OS === "ios" ? "padding" : undefined}
    >
      <ScrollView
        contentContainerStyle={{
          paddingHorizontal: 16,
          paddingVertical: 20,
          alignItems: "center",
        }}
        keyboardShouldPersistTaps="handled"
      >
        <View style={{ width: "100%", maxWidth: 720, gap: 24 }}>
          {/* Section 1: Ingest */}
          <View
            style={{
              backgroundColor: "#ffffff",
              padding: 16,
              borderRadius: 8,
              shadowColor: "#000",
              shadowOpacity: 0.05,
              shadowRadius: 4,
              elevation: 2,
            }}
          >
            <Text style={{ fontSize: 18, fontWeight: "700", marginBottom: 8 }}>
              1. Ingest document
            </Text>
            <Text style={{ marginBottom: 8, color: "#555" }}>
              Paste text here and store it in the RAG store.
            </Text>

            <Text style={{ fontWeight: "600", marginBottom: 4 }}>Source</Text>
            <TextInput
              value={docSource}
              onChangeText={setDocSource}
              placeholder="e.g. official_website, blog, note"
              style={{
                borderWidth: 1,
                borderColor: "#ccc",
                borderRadius: 6,
                paddingHorizontal: 8,
                paddingVertical: 6,
                marginBottom: 10,
                backgroundColor: "#fff",
              }}
            />

            <Text style={{ fontWeight: "600", marginBottom: 4 }}>
              Document text
            </Text>
            <TextInput
              value={docText}
              onChangeText={setDocText}
              multiline
              numberOfLines={6}
              placeholder="Paste text here..."
              style={{
                borderWidth: 1,
                borderColor: "#ccc",
                borderRadius: 6,
                paddingHorizontal: 8,
                paddingVertical: 8,
                minHeight: 120,
                textAlignVertical: "top",
                backgroundColor: "#fff",
              }}
            />

            <View style={{ marginTop: 12 }}>
              <Button
                title={ingestLoading ? "Ingesting..." : "Ingest into RAG"}
                onPress={handleIngest}
                disabled={ingestLoading}
              />
            </View>

            {ingestStatus && (
              <Text style={{ marginTop: 8, color: "#333" }}>
                {ingestStatus}
              </Text>
            )}
          </View>

          {/* Section 2: Query */}
          <View
            style={{
              backgroundColor: "#ffffff",
              padding: 16,
              borderRadius: 8,
              shadowColor: "#000",
              shadowOpacity: 0.05,
              shadowRadius: 4,
              elevation: 2,
            }}
          >
            <Text style={{ fontSize: 18, fontWeight: "700", marginBottom: 8 }}>
              2. Ask a question
            </Text>
            <Text style={{ marginBottom: 8, color: "#555" }}>
              Ask and the backend will search your stored chunks with
              Ollama embeddings and answer using the local model.
            </Text>

            <Text style={{ fontWeight: "600", marginBottom: 4 }}>
              Question
            </Text>
            <TextInput
              value={question}
              onChangeText={setQuestion}
              placeholder=""
              style={{
                borderWidth: 1,
                borderColor: "#ccc",
                borderRadius: 6,
                paddingHorizontal: 8,
                paddingVertical: 8,
                backgroundColor: "#fff",
                marginBottom: 10,
              }}
            />

            <View style={{ marginTop: 4 }}>
              <Button
                title={queryLoading ? "Asking..." : "Ask RAG"}
                onPress={handleQuery}
                disabled={queryLoading}
              />
            </View>

            {error && (
              <Text style={{ marginTop: 8, color: "red" }}>{error}</Text>
            )}

            {answer ? (
              <View style={{ marginTop: 12 }}>
                <Text style={{ fontWeight: "700", marginBottom: 4 }}>
                  Answer
                </Text>
                <Text style={{ color: "#222" }}>{answer}</Text>
              </View>
            ) : null}

            {chunks.length > 0 && (
              <View style={{ marginTop: 16 }}>
                <Text style={{ fontWeight: "700", marginBottom: 4 }}>
                  Retrieved context ({chunks.length})
                </Text>
                {chunks.map((c, idx) => (
                  <View
                    key={c.id + "-" + c.chunk_index + "-" + idx}
                    style={{
                      marginBottom: 8,
                      padding: 8,
                      borderRadius: 6,
                      backgroundColor: "#f0f0f0",
                    }}
                  >
                    <Text
                      style={{
                        fontSize: 12,
                        color: "#666",
                        marginBottom: 4,
                      }}
                    >
                      {c.source ?? "unknown"} · chunk {c.chunk_index ?? "?"}
                      {typeof c.distance === "number"
                        ? ` · dist: ${c.distance.toFixed(3)}`
                        : ""}
                    </Text>
                    <Text style={{ color: "#222" }}>{c.text}</Text>
                  </View>
                ))}
              </View>
            )}
          </View>
        </View>
      </ScrollView>
    </KeyboardAvoidingView>
  );
}
