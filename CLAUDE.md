# Projekt: porównanie frameworków agentowych dla mniejszych modeli językowych uruchamianych lokalnie

## Kontekst
- Przedmiot: Projekt badawczy. Zespół: Jakub Cholewiński, Sebastian Hetel. Opiekun: dr inż. Piotr Pałka.
- Plan badań przedstawiony 09.11.2026, szablon artykułu (IMRAD) wypełniony. Kodu jeszcze nie ma, więc zaczynamy od pilotażu.
- Harmonogram: listopad 2026: środowisko, pilotaż, implementacja; grudzień 2026: eksperymenty, analiza, raport.
- Tytuł roboczy: "Porównanie frameworków agentowych dla mniejszych modeli językowych uruchamianych lokalnie".

## Cel
Ustalić, czy i na ile wybór frameworka (i rodzaj orkiestracji) wpływa na poprawność, koszt i stabilność wyników systemu agentowego z mniejszym modelem uruchamianym lokalnie.

Luka badawcza: wcześniejsze porównania wykonano na dużych modelach przez API (Rasheed i in. 2026, arXiv:2604.16646: 22 frameworki, zbliżona dokładność, słabsze przegrywają przez błędy orkiestracji; Lwin i Kumar 2026, arXiv:2605.09894: orkiestracja jawna ma podobną poprawność co delegowana i zużywa do 3,5x mniej tokenów). Zespół z poprzedniego semestru dostał 100% na HumanEval we wszystkich wariantach, dlatego bierzemy trudniejszy zbiór: LiveCodeBench.

Hipotezy (do weryfikacji):
- H1, orkiestracja jawna: poprawność zbliżona między frameworkami, różnice w tokenach i czasie.
- H2, orkiestracja delegowana: implementacja frameworka ma większy wpływ, zwłaszcza na stabilność i koszt.

## Ustalone decyzje

### Stałe
- Jeden model: Qwen3.8-27B, kwantyzacja 4-bit (~17 GB). Rozmiar modelu nie jest przedmiotem badania.
- Serwowanie przez vLLM z API zgodnym z OpenAI. Wszystkie frameworki łączą się z tym samym endpointem.
- GPU: RTX 4090, 24 GB VRAM. Na KV cache zostaje ~7 GB, więc kontekst jest ograniczony, a przebiegi idą raczej po kolei.
- Infrastruktura: wszystko uruchamiamy na wynajętej instancji Vast.ai (vLLM, proxy, runner, piaskownica, ocena). Lokalnie (Mac) tylko piszemy kod; nic nie zakładamy o uruchamianiu lokalnym. Kod trafia na instancję przez git. Instancja Vast.ai sama jest kontenerem Dockera, więc piaskownica nie korzysta z Dockera.
- Środowisko Pythona: `uv` z lockfile'em (`uv.lock`), Python 3.12, wersje przypięte dokładnie (`==`) w `pyproject.toml`. Wszystkie frameworki mieszczą się w jednym środowisku (sprawdzone 23.11.2026), w najnowszych wersjach: langgraph 1.2.24, langchain-openai 1.8.5, crewai 1.15.44, agent-framework-core 1.26.0 (+ agent-framework-openai 1.18.2, agent-framework-orchestrations 1.4.2), google-adk 2.15.0, litellm 1.110.2. CrewAI ogranicza pydantic (<2.13), a frameworki ogranicza openai (<3), więc te dwa są niżej niż najnowsze. Z MAF instalujemy tylko core, openai i orchestrations, nie metapakiet `agent-framework` (ciągnie ~30 integracji, głównie bety). vLLM ma osobne środowisko na instancji (krok 2). Zmiana zależności: `uv add --no-sync --bounds exact <pakiet>`, na instancji `uv sync --frozen`.
- Te same role, prompty i narzędzia we wszystkich konfiguracjach. Limit 3 iteracji poprawek.
- Tryb myślenia modelu: na razie wyłączony; ostateczna decyzja po pilotażu (potem stały we wszystkich konfiguracjach).
- Temperatura: w pilotażu nieistotna; ustalamy ją sami przed eksperymentem głównym (3 powtórzenia na 3 seedach).
- Bez testów w języku polskim. Bez wariantu bazowego w czystym Pythonie.

### Architektura (identyczna w każdym frameworku)
- Koordynator: planuje zadanie i zleca pracę Wykonawcy.
- Wykonawca: pisze program i sam sprawdza go na przykładach z treści. Narzędzia: `run_python`, `run_tests`.
- Weryfikator (wariant roboczy, sens sprawdzamy w pilotażu): niezależnie od Wykonawcy sprawdza program i zwraca werdykt w JSON (schemat do ustalenia, np. `{"verdict": "pass" | "fail", "feedback": "..."}`). Ma te same narzędzia co Wykonawca (`run_python`, `run_tests`); nie wierzy Wykonawcy na słowo, uruchamia wszystko sam. Robi trzy rzeczy:
  1. Przykłady z treści (`public_test_cases`): uruchamia i porównuje z oczekiwanym wyjściem (wynik pewny).
  2. Testy odporności: sam tworzy wejścia (przypadki brzegowe, maksymalny rozmiar danych; duże wejścia generuje skryptem przez `run_python`). Sprawdza tylko, czy program się nie wywraca, mieści w limicie czasu i ma poprawny format wyjścia. NIE podaje oczekiwanego wyjścia (unikamy fałszywych `fail` z błędnie wyliczonych wyników).
  3. Przegląd „z grubsza poprawny”: czyta kod względem treści (złożoność vs ograniczenia, przypadki szczególne z treści).
  - `fail` z twardych powodów: błąd na przykładzie, wywrotka, przekroczony czas. Z przeglądu `fail` tylko przy wskazaniu konkretnego problemu (np. O(n²) przy n = 2·10⁵), nie z ogólnych wątpliwości.
  - Świadome ograniczenie: Weryfikator nie wykryje złej odpowiedzi w przypadku brzegowym, gdy program działa, ale liczy źle.
  - Inspiracja: AgentCoder (arXiv:2312.13010, niezależny projektant testów). Odrzucone: same przykłady z treści (Weryfikator dublowałby Wykonawcę) i własne testy z oczekiwanym wyjściem (ryzyko błędnych wyników, por. Reflexion arXiv:2303.11366, MapCoder ACL 2024).
- Werdykt `pass` kończy przebieg, `fail` oznacza poprawkę; maksymalnie 3 iteracje.
- Iteracja = jedno przekazanie kodu Wykonawcy do Weryfikatora. Liczymy ją sami z logów przebiegu, nie z liczników frameworka.
- Twarde limity na przebieg (konfigurowalne): limit czasu na start 10 min (rozważamy 5 min) i maksymalna liczba zapytań do modelu. Przekroczenie zapisujemy jako status `timeout` (wynik konfiguracji, nie awaria infrastruktury).
- Agenci widzą tylko testy publiczne. Ukryte testy służą wyłącznie do oceny końcowej.

### Konfiguracje
| # | Framework | Tryb | Orkiestracja |
|---|-----------|------|--------------|
| 1 | LangGraph | graf stanów | jawna (kod steruje) |
| 2 | Microsoft Agent Framework | workflow | jawna |
| 3 | CrewAI | proces hierarchiczny | delegowana (model-menedżer steruje) |
| 4 | Microsoft Agent Framework | magentic | delegowana |
| 5 | Google ADK | opcjonalnie, decyzja po pilotażu | delegowana |

Microsoft Agent Framework w dwóch trybach to porównanie kontrolne: ten sam framework, inny sposób sterowania.

### Zadania
- LiveCodeBench (LeetCode, AtCoder, Codeforces), poziomy easy / medium / hard.
- Badanie główne: 15 zadań na poziom medium i hard. Poziom easy dołączamy tylko wtedy, gdy medium i hard okażą się za trudne.
- Nie filtrujemy zadań po dacie ani pod kątem tego, czy model je widział. Badamy system wieloagentowy (przepływ informacji, orkiestrację), a nie wiedzę modelu.
- 3 powtórzenia każdego zadania w każdej konfiguracji: 4–5 konfiguracji × 30 zadań × 3 = 360–450 przebiegów.

### Metryki (na przebieg)
- Poprawność: czy program końcowy przechodzi wszystkie ukryte testy.
- Stabilność (powtarzalność): czy wynik tego samego zadania jest zgodny w 3 powtórzeniach.
- Tokeny (prompt i completion) oraz czas na zadanie.
- Liczba iteracji poprawek.
- Błędy wywołań narzędzi i błędy formatu JSON.
- Trafność werdyktów Weryfikatora względem ukrytych testów (macierz: `pass`/`fail` Weryfikatora × poprawny/niepoprawny program), dla każdego werdyktu w przebiegu.

### Ryzyka
- Google ADK jest projektowany pod Gemini. Tłumaczenie na format OpenAI może psuć JSON i wywołania narzędzi. Testujemy to w pilotażu; jeśli nie zadziała, ADK wypada (to też jest wynik).
- VRAM: ograniczamy kontekst i liczbę równoległych przebiegów.
- Za trudne zadania: oceniamy w pilotażu, ewentualnie dodajemy poziom easy.
- Frameworki delegowane mają wbudowane prompty, których nie wyrównamy: analizujemy logi i opisujemy to w ograniczeniach.

## OBECNY ETAP: PILOTAŻ

### Cele pilotażu
1. Uruchomić vLLM z modelem i sprawdzić, że tool calling działa (zużycie VRAM, długość kontekstu).
2. Zbudować wspólne elementy (runner, piaskownica, narzędzia, ocena, logowanie metryk) i 4–5 konfiguracji.
3. Sprawdzić, czy każdy framework poprawnie wywołuje narzędzia z modelem lokalnym i utrzymuje format JSON.
4. Zmierzyć czas przebiegu i na tej podstawie oszacować czas eksperymentu głównego.
5. Ocenić trudność poziomów: czy medium i hard nie dają wyników bliskich 0%.
6. Zdecydować, czy Google ADK zostaje w badaniu.
7. Sprawdzić, czy wariant Weryfikatora ma sens: jak często jego werdykt jest błędny względem ukrytych testów i czy pętla poprawek w ogóle się uruchamia.

### Zakres
- 2 zadania na poziom (easy, medium, hard), razem 6 zadań. Zadania z pilotażu mogą wejść do próby głównej.
- 1 przebieg na konfigurację; opcjonalnie 3, żeby zobaczyć rozrzut.

### Kryteria zakończenia pilotażu
- Każda konfiguracja przechodzi przez 6 zadań bez awarii infrastruktury.
- Każdy przebieg ma zapisane wszystkie metryki.
- Podjęte decyzje: ADK tak/nie, poziom easy tak/nie, długość kontekstu, tryb myślenia modelu, limit czasu przebiegu, realny czas eksperymentu głównego. (Temperatura dopiero przed eksperymentem głównym.)
- Analizą statystyczną zajmujemy się dopiero w kolejnym etapie.

## Propozycje techniczne (do sprawdzenia, jeszcze NIE ustalone)
- **LiveCodeBench:** zbiór na Hugging Face `livecodebench/code_generation_lite` (najnowszy release). Pola to m.in. `platform`, `difficulty`, `contest_date`, `starter_code`, `public_test_cases`, `private_test_cases` (zakodowane). Uwaga: zadania LeetCode są funkcyjne (starter code, `class Solution`), a AtCoder i Codeforces działają na stdin/stdout. `run_tests` i ocena muszą obsłużyć oba typy, albo w pilotażu bierzemy tylko zadania stdin; do decyzji. Do oceny ukrytymi testami można wykorzystać kod ewaluacji z repozytorium LiveCodeBench. Ocenę sprawdzić na ręcznie napisanym poprawnym rozwiązaniu.
- **vLLM:** tool calling wymaga `--enable-auto-tool-choice` i właściwego `--tool-call-parser` (dla Qwen zwykle `hermes`; sprawdzić dla tego modelu). `--max-model-len` dobrać pod VRAM. Ustalić format kwantyzacji (AWQ, GPTQ itp.). Przy włączonym parserze rozumowania sprawdzić, czy frameworki poprawnie obsługują pole `reasoning_content`.
- **Tokeny (USTALONE):** liczymy jednolicie przez cienkie proxy logujące przed vLLM (każde zapytanie: `run_id`, tokeny prompt/completion, czas), a nie licznikami frameworków. `run_id` przekazujemy w ścieżce adresu: `base_url = http://<proxy>/run/<run_id>/v1`, więc żaden framework nie wymaga zmian. Proxy przepuszcza streaming i dopisuje `stream_options.include_usage`.
- **Piaskownica (USTALONE):** `run_python` / `run_tests` jako subprocess z timeoutem i limitami zasobów w katalogu tymczasowym (bez Dockera, bo instancja Vast.ai to już kontener).
- **Wspólny interfejs:** każda konfiguracja implementuje `run(task, config) -> RunResult` (kod końcowy, historia werdyktów, liczba iteracji, wywołania narzędzi, błędy). Runner CLI: `--config --tasks --reps`. Wyniki zapisywane jako JSONL.
- **Prompty ról:** w jednym module, importowane przez wszystkie implementacje.
- **Konfiguracje delegowane:** w CrewAI i MAF magentic rolę Koordynatora pełni menedżer frameworka. Do decyzji: wbudowany menedżer czy własny z naszym promptem. Limit 3 iteracji egzekwować ustawieniami frameworka (maks. liczba iteracji / rund); sprawdzić w dokumentacji.
- **Połączenie z vLLM:** LangGraph przez `ChatOpenAI(base_url=...)`, CrewAI przez LLM z prefiksem `openai/` i `base_url`, MAF przez klienta OpenAI z `base_url`, ADK przez `LiteLlm`. API frameworków szybko się zmienia, więc sprawdzać aktualną dokumentację i przypiąć wersje.

### Proponowana struktura repo
    agent-bench/
      CLAUDE.md
      configs/            # model, URL vLLM, limity, temperatura
      data/               # ID zadań pilotażowych i próby głównej
      bench/
        tasks.py          # ładowanie LiveCodeBench, wybór zadań
        sandbox.py        # run_python, run_tests
        grader.py         # ocena na ukrytych testach
        prompts.py        # wspólne prompty ról
        proxy.py          # logowanie tokenów i czasu przed vLLM
        runner.py         # CLI
      frameworks/
        langgraph_explicit.py
        maf_workflow.py
        crewai_hierarchical.py
        maf_magentic.py
        adk_delegated.py
      results/            # JSONL z przebiegów
      analysis/

## Kolejność pracy
1. ✅ Szkielet repo i środowisko Pythona, z przypiętymi wersjami frameworków (`pyproject.toml`, `uv.lock`, `configs/pilot.yaml` + `bench/config.py`, `scripts/setup_vastai.sh`). Repo jest na razie tylko lokalne (brak zdalnego na GitHubie).
2. Skrypt startu vLLM i test tool callingu (jedno zapytanie z jednym narzędziem).
3. Ładowanie LiveCodeBench, wybór 6 zadań pilotażowych, ocena na ukrytych testach.
4. Piaskownica, narzędzia agentów, proxy do logowania tokenów.
5. LangGraph jako pierwsza, referencyjna implementacja; potem MAF workflow, CrewAI, MAF magentic, ADK.
6. Przebieg pilotażu i tabela wyników (konfiguracja × zadanie: poprawność, tokeny, czas, iteracje, błędy).
