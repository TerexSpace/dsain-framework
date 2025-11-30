# JMLR Submission Checklist

## Pre-Submission Verification

### Format Requirements
- [x] Document uses jmlr2e.sty style file
- [x] Single-column format, 11pt font
- [x] Margins: 1.25in left and right
- [x] Abstract present with keywords
- [x] Author information on first page
- [x] \jmlrheading command configured (to be updated by editor)
- [x] \ShortHeadings command set
- [x] \firstpageno command present
- [x] \editor command present (to be filled by editor)

### Content Requirements
- [x] Paper presents new algorithms with theoretical justification
- [x] Experimental results with empirical validation
- [x] Clear articulation of contributions
- [x] Proper literature review and positioning
- [x] Discussion of limitations and future work

### Technical Completeness
- [x] All mathematical notation defined
- [x] Proofs provided for theorems
- [x] Assumptions clearly stated
- [x] Convergence analysis included
- [x] Privacy guarantees formally stated

### Citation Format
- [x] Using natbib citation style (included in jmlr2e.sty)
- [x] \citep{} for parenthetical citations
- [x] \citet{} for textual citations
- [x] Citations are not used as nouns

### Reproducibility
- [x] Algorithm pseudocode provided
- [x] Hyperparameters specified
- [x] Random seeds documented
- [x] Implementation code available
- [x] Data generation procedure described

### Figures and Tables
- [x] All figures/tables have captions
- [x] Tables use booktabs style
- [x] Figures are vector graphics where possible
- [x] Figure references capitalized (Figure 1, Table 1)

## Submission Process

### Required Documents
1. [ ] PDF of manuscript (main.pdf)
2. [ ] LaTeX source files (main.tex, jmlr2e.sty, references.bib)
3. [ ] Supplementary materials (code, appendices)
4. [ ] Cover letter

### Submission Portal
- URL: http://jmlr.csail.mit.edu/manudb
- Create account if not existing
- Upload manuscript and supplementary materials

### Cover Letter Should Include
- [ ] Brief summary of contributions
- [ ] Statement of originality (not published elsewhere)
- [ ] Declaration of conflicts of interest
- [ ] Suggested action editors (3-5)
- [ ] Suggested reviewers (3-5)
- [ ] Confirmation all co-authors consent to submission

### Suggested Action Editors
Based on paper content (federated learning, Byzantine resilience, privacy):
1. [Editor Name 1] - Expertise in distributed learning
2. [Editor Name 2] - Expertise in privacy-preserving ML
3. [Editor Name 3] - Expertise in optimization
4. [Editor Name 4] - Expertise in robust statistics
5. [Editor Name 5] - Expertise in applied ML systems

### Funding and Conflicts Disclosure
Required categories to address:
- [ ] Third-party funding received in last 36 months
- [ ] Financial relationships with entities relevant to work
- [ ] Hardware donations or cloud computing services
- [ ] Employment/sabbaticals at relevant companies

## Post-Acceptance Checklist

### Final Version
- [ ] Add page numbers (assigned by editor)
- [ ] Add publication date (assigned by editor)
- [ ] Update \jmlrheading with final information
- [ ] Create HTML version (encouraged)
- [ ] Submit source code to UCI ML Repository (encouraged)

### Copyright
- [ ] Sign Permission to Publish form
- [ ] Sign Software Release form (if code included)

### File Naming Convention
Following JMLR convention (assuming first author "Smith" and year 2025):
- smith25a.pdf - Main paper
- smith25a.tex - LaTeX source
- smith25a-appendix.pdf - Online appendix (if any)
- smith25a-code.tar.gz - Source code

## Quality Self-Assessment

### Novelty
- Does the paper present sufficiently novel contributions? [YES]
- Are contributions clearly differentiated from prior work? [YES]

### Technical Soundness
- Are all proofs correct and complete? [VERIFY]
- Are experimental results reproducible? [YES]
- Are baselines appropriate and fairly compared? [YES]

### Clarity
- Is the paper well-written and easy to follow? [VERIFY]
- Are all terms defined before use? [YES]
- Is notation consistent throughout? [YES]

### Impact
- Does the work advance the field? [YES]
- Will it be cited by future researchers? [EXPECTED]
- Does it open new research directions? [YES]

## Common JMLR Formatting Issues to Avoid

1. ❌ Using i.e. and e.g. incorrectly
   ✅ Use "that is" and "for example" or proper punctuation

2. ❌ Treating citations as nouns: "Using (Smith, 1999), we..."
   ✅ Correct: "Using the method of Smith (1999), we..."

3. ❌ Forgetting periods in equations at sentence ends
   ✅ Equations in sentences need proper punctuation

4. ❌ Using nonstandard fonts
   ✅ Stick to Times Roman / Computer Modern

5. ❌ Numbering all equations
   ✅ Only number equations referenced elsewhere

6. ❌ Using dashes incorrectly
   ✅ Use --- for em-dashes with no spaces

## Status

| Item | Status |
|------|--------|
| Manuscript complete | ✅ |
| Bibliography verified | ✅ |
| Code tested | ✅ |
| Proofs reviewed | 🔄 Pending final review |
| Co-author approval | ⏳ TBD |
| Submission ready | ⏳ After final review |
