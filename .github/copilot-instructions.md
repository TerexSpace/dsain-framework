# JMLR Manuscript: DSAIN Framework

## Project Overview
Academic paper submission for **Journal of Machine Learning Research (JMLR)** presenting DSAIN (Distributed Sovereign AI Network), a federated learning framework with Byzantine-resilient aggregation and differential privacy for emerging economies. Target: Kazakhstan's Alem AI Center deployment case study.

## Architecture

**Two-tier structure:**
- `latex/`: Academic manuscript (main.tex + jmlr2e.sty + references.bib)
- `code/`: Reference implementation (dsain.py)

The paper presents three core algorithms: **FedSov** (communication-efficient federated learning), **ByzFed** (Byzantine-resilient aggregation), **blockchain-based provenance system**.

## LaTeX Workflow

### Compilation Commands
Standard JMLR compilation (run from `latex/` directory):
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or with latexmk: `latexmk -pdf main.tex`

### JMLR Style Requirements
- **MUST use jmlr2e.sty** - included in latex/ directory
- Single-column, 11pt, 1.25in margins
- Citation style: `\citep{}` for parenthetical, `\citet{}` for textual
- Never use citations as nouns: ❌ "(Smith, 1999) proposes..." → ✅ "Smith (1999) proposes..."
- Figure/Table references: capitalize (Figure 1, Table 1)
- Em-dashes: `---` with no spaces
- Number only referenced equations

### Custom Macros (defined in main.tex)
```latex
\DSAIN{}      % Algorithm name
\FedSov{}     % FedSov algorithm
\ByzFed{}     % ByzFed aggregation
\dataset      % Dataset symbol
\model        % Model symbol
\loss         % Loss symbol
\expect       % Expectation operator
\grad         % Gradient operator
```

**Always use these macros** for consistency - don't write "DSAIN" or "FedSov" as plain text.

## Python Code Structure

`code/dsain.py` is **reference implementation** matching Algorithm 1 & 2 from paper.
`code/blockchain_provenance.py` implements the blockchain-based model provenance system from Section 4.

### Core Classes (dsain.py)
- `FedSovConfig`: Configuration dataclass
- `GradientCompressor`: Top-k sparsification with error feedback
- `DifferentialPrivacy`: (ε,δ)-DP with Gaussian mechanism
- `ByzFed`: Geometric median aggregation with reputation scoring
- `LocalClient`: Simulated federated learning participant
- `FedSov`: Main orchestrator

### Blockchain Components (blockchain_provenance.py)
- `ModelCommitment`: Cryptographic commitment to model states
- `ProofOfTraining`: Proof-of-Training consensus mechanism
- `Block`: Individual blockchain block
- `ProvenanceBlockchain`: Full blockchain with verification protocols

### Running Experiments
```bash
cd code
pip install numpy scipy matplotlib
python dsain.py --num_clients 100 --num_rounds 200 --byzantine_frac 0.1
```

**Experiment modes:**
- `--mode single`: Single run with specified parameters (generates `convergence_curves.pdf`)
- `--mode byzantine`: Byzantine resilience experiment (generates `byzantine_resilience.pdf`)
- `--mode scalability`: Scalability test with varying client counts (generates `scalability.pdf`)
- `--mode all`: Run all experiments

**Key parameters:**
- `--byzantine_frac`: Fraction of malicious clients (must be < 0.33 per Assumption 1)
- `--compression_ratio`: Top-k sparsification ratio (default 0.1 = 10% of gradients)
- `--seed`: Random seed for reproducibility (default 42)
- `--output_dir`: Output directory for figures (default ../figures)

**Blockchain provenance demo:**
```bash
python blockchain_provenance.py
```
Generates `provenance_chain.json` with complete training history verification.

## Mathematical Notation Conventions

**From Section 3 (Problem Formulation):**
- $n$: total participants
- $f$: Byzantine (malicious) participants, where $f < n/3$ (Assumption 1)
- $\mathcal{H}$: set of honest participants
- $F(\mathbf{w})$: global objective (Equation 1)
- $F_i(\mathbf{w})$: local objective for participant $i$

**Assumptions 1-4:**
1. Byzantine fraction: $f < n/3$
2. $L$-smoothness
3. Bounded variance $\sigma^2$
4. Bounded heterogeneity $\zeta^2$
5. Bounded gradient norm $G$

**Key Theorems:**
- Theorem 1 (Byzantine Resilience): Error bound with filtering
- Theorem 2 (Privacy): (ε,δ)-DP guarantee with composition
- Theorem 3 (Non-Convex Convergence): $\mathcal{O}(1/\sqrt{T})$ rate
- Theorem 4 (Strongly Convex): $\mathcal{O}(1/T)$ rate

## Bibliography Management

`latex/references.bib` uses natbib format:
- Include DOIs or stable URLs for all references
- Author names: use `{\"u}` for umlauts, `{\v{c}}` for carons
- Verify publisher/venue names match official style
- JMLR prioritizes conference proceedings/journal articles over arXiv-only papers

**When adding citations:**
1. Search Google Scholar for correct BibTeX
2. Verify year/venue in official source
3. Add entry to references.bib
4. Use `\citep{key}` in text

## Submission Checklist

See `SUBMISSION_CHECKLIST.md` for full pre-submission verification. Critical items:

**Format:**
- [ ] Document compiles cleanly (no warnings)
- [ ] All figures/tables have captions
- [ ] Math notation consistent throughout
- [ ] Abstract has keywords section

**Content:**
- [ ] Proofs in appendices (app:byzantine, app:convergence)
- [ ] Experimental results with error bars (3 runs minimum)
- [ ] Code available (dsain.py)
- [ ] Case study section complete (Section 6)

**JMLR-specific:**
- [ ] `\jmlrheading` command present (will be filled by editor)
- [ ] `\ShortHeadings` set for running header
- [ ] `\editor` command present
- [ ] Funding/conflicts disclosure in acknowledgments

## Cover Letter Workflow

`latex/cover_letter.tex` must accompany submission. Required sections:

**Key elements:**
1. **Brief contribution summary** (2-3 sentences): State what DSAIN does and why it matters
2. **Originality statement**: Confirm not published/submitted elsewhere
3. **Suggested Action Editors** (3-5 names):
   - Focus on: federated learning, Byzantine resilience, privacy-preserving ML, distributed optimization
   - Check current JMLR editorial board: http://jmlr.org/editorial-board.html
4. **Suggested Reviewers** (3-5):
   - Experts in federated learning systems, Byzantine fault tolerance, differential privacy
   - Include affiliations and why they're qualified
5. **Conflicts of Interest**: Declare any financial/collaborative relationships
6. **Co-author consent**: All authors approve submission

**Template structure:**
```latex
\documentclass[11pt]{letter}
\usepackage{hyperref}
\begin{document}
\begin{letter}{Editor-in-Chief\\Journal of Machine Learning Research}
\opening{Dear Editor-in-Chief,}

[Contribution summary paragraph]

[Originality and scope paragraph]

\textbf{Suggested Action Editors:}
\begin{itemize}
    \item [Name] ([Affiliation]) - Expertise in [area]
    \item ...
\end{itemize}

[Conflicts/funding disclosure]

\closing{Sincerely,\\[Author Names]}
\end{letter}
\end{document}
```

## Hyperparameter Tuning Guide

**Compression Ratio vs Communication Cost:**
- `compression_ratio=0.1` (10%): 78% communication reduction vs FedAvg, minimal accuracy loss (<1%)
- `compression_ratio=0.05` (5%): 89% reduction, 2-3% accuracy loss
- **Recommendation**: Start with 0.1, reduce only if bandwidth extremely limited

**Privacy Budget (ε) vs Accuracy:**
From Table 1 (CIFAR-10, α=0.5):
- ε=∞ (no DP): 90.5% accuracy
- ε=4: 88.1% accuracy (2.4% loss) - **recommended for most deployments**
- ε=2: 85.3% accuracy (5.2% loss)
- ε=1: 81.7% accuracy (8.8% loss) - use only for highly sensitive data

**δ parameter:** Set to 1/n² where n=dataset size. Typical: 10⁻⁵ to 10⁻⁶

**Local Epochs (E) vs Client Drift:**
- E=1: No drift, high communication cost
- E=5: Good balance (default)
- E=10: Significant drift with heterogeneous data (α<0.5), faster convergence with IID data

**Heterogeneity (Dirichlet α):**
- α=0.1: Highly non-IID (worst case) - increase local epochs
- α=0.5: Moderate heterogeneity (realistic) - default settings work well
- α=1.0: Nearly IID - can reduce communication rounds by 20-30%

**Byzantine Tolerance:**
- Max safe fraction: f < n/3 (proven bound)
- Practical recommendation: Design for f ≤ 0.2 (20%) for robust performance
- ByzFed threshold τ=3.0 balances false positives/negatives

## Common Patterns

### Adding New Experiments
1. Update Section 5 (Experiments) with new subsection
2. Add table/figure using `booktabs` package
3. Update `code/dsain.py` with experiment function
4. Report mean ± std over 3+ runs
5. Add comparison to existing baselines (FedAvg, FedProx, SCAFFOLD, Krum, Trimmed Mean)

### Extending Algorithms
When modifying FedSov/ByzFed:
1. Update algorithm pseudocode (Algorithm 1 or 2)
2. Add/modify theorem statement
3. Provide proof sketch in main text + full proof in appendix
4. Update `code/dsain.py` implementation
5. Re-run experiments to verify convergence rates

### Adding References
New citations should follow existing style:
```bibtex
@article{author2025title,
  title={Descriptive Title in Title Case},
  author={Last, First and Last, First},
  journal={Full Journal Name},
  volume={XX},
  pages={XX--XX},
  year={2025}
}
```

## Project-Specific Conventions

**Code Style:**
- Type hints required for all function signatures
- Docstrings in Google format
- NumPy for all numerical operations (not PyTorch/TensorFlow - keeps dependencies minimal)
- Logging via standard library `logging` module

**LaTeX Style:**
- Use `\citep` not `\cite` for consistency
- Inline math: `$...$`; display math: `\begin{equation}...\end{equation}`
- Algorithm blocks: `\begin{algorithm}[t]` with `[t]` placement
- Section references: `Section~\ref{sec:label}` with tilde for non-breaking space

**Variable Naming:**
- Client IDs: integer indices 0 to n-1
- Model parameters: `\mathbf{w}` (bold lowercase for vectors)
- Gradients: `\mathbf{g}` or `\grad` macro
- Updates/deltas: `\Delta_i^t` (capital delta, subscript i, superscript t)

## Key Files

- `latex/main.tex` - Main manuscript (approx. 10k lines compiled)
- `latex/references.bib` - 272 lines, all verified citations
- `latex/cover_letter.tex` - Cover letter template for submission
- `code/dsain.py` - 600+ lines, executable reference implementation with visualization
- `code/blockchain_provenance.py` - 400+ lines, blockchain provenance system
- `figures/` - Generated plots and exported blockchain data
- `SUBMISSION_CHECKLIST.md` - Pre-submission verification items
- `README.md` - Project overview and quick-start guide

## Author Information

**Primary Author:** Almas Ospanov  
**Affiliation:** L.N. Gumilev Eurasian National University, Astana, Kazakhstan  
**Email:** ospanov_ad_4@enu.kz  
**ORCID:** https://orcid.org/0009-0004-3834-130X
