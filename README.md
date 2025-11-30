# DSAIN: Sovereign Federated Learning with Byzantine-Resilient Aggregation

## Paper Information

**Title:** Sovereign Federated Learning with Byzantine-Resilient Aggregation: A Framework for Decentralized AI Infrastructure in Emerging Economies

**Target Journal:** Journal of Machine Learning Research (JMLR)

**Paper ID:** [To be assigned upon submission]

## Repository Structure

```
jmlr_manuscript/
├── latex/
│   ├── main.tex           # Main LaTeX manuscript
│   ├── references.bib     # Bibliography file
│   └── jmlr2e.sty         # JMLR style file
├── code/
│   └── dsain.py           # Reference implementation
├── figures/
│   └── [Generated during compilation]
├── appendices/
│   └── [Additional proofs and derivations]
├── data/
│   └── [Synthetic data generation scripts]
└── README.md              # This file
```

## Compilation Instructions

### Requirements

- TeXLive 2020+ or MiKTeX
- Python 3.8+ (for code verification)
- NumPy, SciPy (for experiments)

### LaTeX Compilation

```bash
cd latex
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or using latexmk:

```bash
cd latex
latexmk -pdf main.tex
```

### Code Execution

```bash
cd code
pip install numpy scipy
python dsain.py --num_clients 100 --num_rounds 200 --byzantine_frac 0.1
```

## Key Contributions

1. **FedSov Algorithm**: Communication-efficient federated learning with adaptive gradient compression achieving O(1/√T) convergence for non-convex objectives.

2. **ByzFed Aggregation**: Byzantine-resilient aggregation mechanism tolerating up to ⌊(n-1)/3⌋ malicious participants with provable robustness guarantees.

3. **Differential Privacy Integration**: (ε, δ)-differential privacy with composition analysis for multi-round federated learning.

4. **Blockchain Provenance**: Lightweight model provenance system using cryptographic commitments for verifiable training history.

5. **Case Study**: National-scale deployment evaluation at Kazakhstan's Alem AI Center.

## Reproducibility Checklist

- [x] All hyper-parameters clearly specified in paper
- [x] Random seeds provided for experiments
- [x] Complete algorithmic descriptions with pseudocode
- [x] Statistical significance measures reported
- [x] Reference implementation provided
- [x] Synthetic data generation code included

## Citation

```bibtex
@article{ospanov2025dsain,
  title={Sovereign Federated Learning with Byzantine-Resilient Aggregation: 
         A Framework for Decentralized AI Infrastructure in Emerging Economies},
  author={Ospanov, Almas and Author Two},
  journal={Journal of Machine Learning Research},
  year={2025},
  volume={XX},
  pages={1--XX}
}
```

## License

- Paper: CC-BY 4.0
- Code: MIT License

## Contact

- Almas Ospanov: ospanov_ad_4@enu.kz (ORCID: https://orcid.org/0009-0004-3834-130X)
- Author Two: author2@institution.org

## Acknowledgments

[To be completed by authors - declare all funding sources and potential conflicts of interest]
