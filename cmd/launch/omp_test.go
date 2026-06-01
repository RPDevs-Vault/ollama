package launch

import (
	"os"
	"path/filepath"
	"runtime"
	"slices"
	"testing"

	modelpkg "github.com/ollama/ollama/types/model"
	"gopkg.in/yaml.v3"
)

func TestOMPIntegration(t *testing.T) {
	o := &OMP{}

	t.Run("String", func(t *testing.T) {
		if got := o.String(); got != "OMP" {
			t.Errorf("String() = %q, want %q", got, "OMP")
		}
	})

	t.Run("implements Runner", func(t *testing.T) {
		var _ Runner = o
	})

	t.Run("implements ManagedSingleModel", func(t *testing.T) {
		var _ ManagedSingleModel = o
	})

	t.Run("implements ManagedModelListConfigurer", func(t *testing.T) {
		var _ ManagedModelListConfigurer = o
	})

	t.Run("does not require interactive onboarding", func(t *testing.T) {
		var _ ManagedInteractiveOnboarding = o
		if o.RequiresInteractiveOnboarding() {
			t.Fatal("OMP onboarding should not require an interactive terminal")
		}
	})
}

func TestOMPArgs(t *testing.T) {
	o := &OMP{}

	tests := []struct {
		name  string
		model string
		args  []string
		want  []string
	}{
		{"with model", "gemma4", nil, []string{"--model", "ollama/gemma4"}},
		{"with cloud model", "kimi-k2.6:cloud", nil, []string{"--model", "ollama/kimi-k2.6:cloud"}},
		{"empty model", "", nil, nil},
		{"with model and extra", "gemma4", []string{"--help"}, []string{"--model", "ollama/gemma4", "--help"}},
		{"already qualified", "ollama/gemma4", nil, []string{"--model", "ollama/gemma4"}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := o.args(tt.model, tt.args)
			if !slices.Equal(got, tt.want) {
				t.Errorf("args(%q, %v) = %v, want %v", tt.model, tt.args, got, tt.want)
			}
		})
	}
}

func TestOMPFindPath(t *testing.T) {
	o := &OMP{}

	t.Run("finds omp in PATH", func(t *testing.T) {
		tmpDir := t.TempDir()
		name := "omp"
		if runtime.GOOS == "windows" {
			name = "omp.exe"
		}
		fakeBin := filepath.Join(tmpDir, name)
		os.WriteFile(fakeBin, []byte("#!/bin/sh\n"), 0o755)
		t.Setenv("PATH", tmpDir)

		got, err := o.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fakeBin {
			t.Errorf("findPath() = %q, want %q", got, fakeBin)
		}
	})

	t.Run("falls back to ~/.local/bin/omp", func(t *testing.T) {
		home := t.TempDir()
		setTestHome(t, home)
		t.Setenv("PATH", t.TempDir())

		fallback := filepath.Join(home, ".local", "bin", ompExecutableNames()[0])
		os.MkdirAll(filepath.Dir(fallback), 0o755)
		os.WriteFile(fallback, []byte("#!/bin/sh\n"), 0o755)

		got, err := o.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fallback {
			t.Errorf("findPath() = %q, want %q", got, fallback)
		}
	})

	t.Run("falls back to ~/.bun/bin/omp", func(t *testing.T) {
		home := t.TempDir()
		setTestHome(t, home)
		t.Setenv("PATH", t.TempDir())

		fallback := filepath.Join(home, ".bun", "bin", ompExecutableNames()[0])
		os.MkdirAll(filepath.Dir(fallback), 0o755)
		os.WriteFile(fallback, []byte("#!/bin/sh\n"), 0o755)

		got, err := o.findPath()
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if got != fallback {
			t.Errorf("findPath() = %q, want %q", got, fallback)
		}
	})

	t.Run("returns error when not found", func(t *testing.T) {
		home := t.TempDir()
		setTestHome(t, home)
		t.Setenv("PATH", t.TempDir())

		if _, err := o.findPath(); err == nil {
			t.Fatal("expected error, got nil")
		}
	})
}

func TestOMPConfigureWithModelsWritesModelsYML(t *testing.T) {
	home := t.TempDir()
	setTestHome(t, home)
	t.Setenv("OLLAMA_HOST", "http://0.0.0.0:11434")

	o := &OMP{}
	models := []LaunchModel{
		{
			Name:            "glm-5.1:cloud",
			ContextLength:   202_752,
			MaxOutputTokens: 131_072,
		},
		{
			Name:         "qwen3.6",
			Capabilities: []modelpkg.Capability{modelpkg.CapabilityVision},
		},
	}
	if err := o.ConfigureWithModels("glm-5.1:cloud", models); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	path := filepath.Join(home, ".omp", "agent", "models.yml")
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("failed to read models.yml: %v", err)
	}

	cfg := parseOMPConfigYAML(t, data)
	provider := ompProviderFromYAML(t, cfg)
	if provider["baseUrl"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("baseUrl = %v, want connectable OpenAI-compatible host", provider["baseUrl"])
	}
	if provider["api"] != "openai-responses" {
		t.Fatalf("api = %v, want openai-responses", provider["api"])
	}
	if provider["auth"] != "none" {
		t.Fatalf("auth = %v, want none", provider["auth"])
	}
	discovery, _ := provider["discovery"].(map[string]any)
	if discovery["type"] != "ollama" {
		t.Fatalf("discovery = %v, want type ollama", discovery)
	}

	entries := ompModelEntriesFromYAML(t, provider)
	if len(entries) != 2 {
		t.Fatalf("models length = %d, want 2", len(entries))
	}
	if entries[0]["id"] != "glm-5.1:cloud" {
		t.Fatalf("first model id = %v, want primary first", entries[0]["id"])
	}
	if got := numericYAMLValue(entries[0]["contextWindow"]); got != 202_752 {
		t.Fatalf("contextWindow = %d, want 202752", got)
	}
	if got := numericYAMLValue(entries[0]["maxTokens"]); got != 131_072 {
		t.Fatalf("maxTokens = %d, want 131072", got)
	}
	if input := stringSliceYAMLValue(entries[1]["input"]); !slices.Equal(input, []string{"text", "image"}) {
		t.Fatalf("vision input = %v, want [text image]", input)
	}
	if got := o.CurrentModel(); got != "glm-5.1:cloud" {
		t.Fatalf("CurrentModel = %q, want glm-5.1:cloud", got)
	}
	if paths := o.Paths(); len(paths) != 1 || paths[0] != path {
		t.Fatalf("Paths = %v, want [%s]", paths, path)
	}
}

func TestOMPConfigureWithModelsPreservesExistingConfig(t *testing.T) {
	home := t.TempDir()
	setTestHome(t, home)

	path := filepath.Join(home, ".omp", "agent", "models.yml")
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	existing := []byte(`
providers:
  anthropic:
    baseUrl: https://example.com/anthropic
  ollama:
    baseUrl: http://old-host:11434
    api: openai-responses
    auth: none
    models:
      - id: old-model
        name: Old Model
        customField: keep-me
`)
	if err := os.WriteFile(path, existing, 0o644); err != nil {
		t.Fatal(err)
	}

	o := &OMP{}
	if err := o.ConfigureWithModels("new-model", []LaunchModel{{Name: "new-model"}, {Name: "old-model"}}); err != nil {
		t.Fatalf("ConfigureWithModels returned error: %v", err)
	}

	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	cfg := parseOMPConfigYAML(t, data)
	providers, _ := cfg["providers"].(map[string]any)
	if _, ok := providers["anthropic"]; !ok {
		t.Fatalf("expected non-Ollama provider to be preserved: %v", providers)
	}

	provider := ompProviderFromYAML(t, cfg)
	if provider["baseUrl"] != "http://127.0.0.1:11434/v1" {
		t.Fatalf("baseUrl = %v, want repaired OpenAI-compatible host", provider["baseUrl"])
	}

	entries := ompModelEntriesFromYAML(t, provider)
	if len(entries) != 2 {
		t.Fatalf("models length = %d, want 2", len(entries))
	}
	if entries[0]["id"] != "new-model" {
		t.Fatalf("first model id = %v, want new-model", entries[0]["id"])
	}
	if entries[1]["id"] != "old-model" {
		t.Fatalf("second model id = %v, want old-model", entries[1]["id"])
	}
	if entries[1]["customField"] != "keep-me" {
		t.Fatalf("custom field was not preserved: %v", entries[1])
	}
}

func parseOMPConfigYAML(t *testing.T, data []byte) map[string]any {
	t.Helper()
	var cfg map[string]any
	if err := yaml.Unmarshal(data, &cfg); err != nil {
		t.Fatalf("generated YAML did not parse: %v\n%s", err, data)
	}
	return cfg
}

func ompProviderFromYAML(t *testing.T, cfg map[string]any) map[string]any {
	t.Helper()
	providers, ok := cfg["providers"].(map[string]any)
	if !ok {
		t.Fatalf("providers missing from config: %v", cfg)
	}
	provider, ok := providers["ollama"].(map[string]any)
	if !ok {
		t.Fatalf("ollama provider missing from config: %v", providers)
	}
	return provider
}

func ompModelEntriesFromYAML(t *testing.T, provider map[string]any) []map[string]any {
	t.Helper()
	rawModels, ok := provider["models"].([]any)
	if !ok {
		t.Fatalf("provider models missing: %v", provider)
	}
	models := make([]map[string]any, 0, len(rawModels))
	for _, raw := range rawModels {
		entry, ok := raw.(map[string]any)
		if !ok {
			t.Fatalf("model entry has unexpected type %T: %v", raw, raw)
		}
		models = append(models, entry)
	}
	return models
}

func numericYAMLValue(value any) int {
	switch v := value.(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	default:
		return 0
	}
}

func stringSliceYAMLValue(value any) []string {
	raw, _ := value.([]any)
	out := make([]string, 0, len(raw))
	for _, item := range raw {
		if s, ok := item.(string); ok {
			out = append(out, s)
		}
	}
	return out
}
