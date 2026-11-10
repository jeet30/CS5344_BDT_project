"""Build an independent, ten-page revision from the original report source."""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'report.tex').read_text()

def between(a, b):
    return source.split(a, 1)[1].split(b, 1)[0]

def figure(name, caption, label, width=r'\linewidth'):
    return rf'\begin{{figure}}[H]\centering\includegraphics[width={width}]{{../figs/{name}.pdf}}\caption{{{caption}}}\label{{{label}}}\end{{figure}}'

preamble = source.split(r'\begin{document}', 1)[0]
preamble += r'''
\usepackage{float}
\hypersetup{pdftitle={Detector-Aware Data Augmentation for Network Intrusion Detection},pdfauthor={Aayush Majumdar and Vadiraj Rao}}
\setlength{\textfloatsep}{9pt}
\setlength{\intextsep}{9pt}
\raggedbottom
\begin{document}
'''
cover = between(r'\begin{titlepage}', r'\end{titlepage}')
cover = cover.replace('Data Augmentation for Anomaly Detection', 'Detector-Aware Data Augmentation for Network Intrusion Detection')
pages = []
pages.append(r'''
\begin{center}
{\Large\bfseries Detector-Aware Data Augmentation for Network Intrusion Detection}\\[4pt]
{\normalsize Conditional Copulas and Mixed-Attribute Clustering in a Shared Processing Pipeline}\\[4pt]
Aayush Majumdar and Vadiraj Rao \quad Group 42, CS5344
\end{center}
\begin{abstract}\noindent
We study the generation of normal-only training data for fixed ECOD and Isolation Forest detectors on two network-intrusion datasets. The supplied NSL-KDD and UNSW-NB15 tables contain 30\,000 and 50\,000 labeled records; the required outputs contain 40\,000 and 70\,000 normal records. Our objective is downstream attack-ranking utility rather than distributional fidelity alone. A shared implementation combines configurable generation, category weighting, duplicate jitter, tail expansion, domain constraints, and feature neutralization. UNSW-NB15 benefits from contrastive normal reweighting and conditional Gaussian copulas with discriminative variance scaling. NSL-KDD benefits from retaining scrubbed normal observations and supplementing them with copulas fitted within SILK-style k-FreqItems clusters. Historical public scores increase from 0.4210 to 0.61452 for UNSW-NB15 and from 0.9292 to 0.93734 for NSL-KDD. Selected submissions achieve final hidden AUPRC of \textbf{0.68508} and \textbf{0.95247}, respectively, on a separate evaluation partition. Negative experiments demonstrate that additional generator complexity or stronger balancing does not reliably improve detector utility. We provide implementation pseudocode, ablations, validation limitations, and lessons for reproducible distribution engineering.
\end{abstract}
\section{Introduction and Research Objectives}
Network-intrusion detection must distinguish legitimate traffic from attacks, including attack types absent from the supplied training data. Here the detector architecture is fixed: our intervention is the normal-only table on which it is trained. The task therefore asks whether changing normal coverage, category prevalence, and feature tails can improve detection without modifying the downstream models.

\textbf{Detector behavior.} ECOD~\cite{ecod} assigns outlier scores using empirical feature-tail probabilities. Isolation Forest~\cite{iforest} uses random feature splits: observations isolated through shorter paths receive higher anomaly scores. Both respond to the distribution of the submitted rows, although neither directly optimizes the attack labels available during augmentation.

\textbf{Evaluation objective.} Let $D=\{(x_i,y_i)\}_{i=1}^{n}$, where $y_i=1$ denotes an attack, and let $G_\theta(D)$ produce $m$ normal training rows. The optimization target is
\begin{equation}
\max_\theta S(\theta),\qquad
S=\tfrac12\left[\operatorname{AP}_{\rm ECOD}+\tfrac1{10}\sum_{r=1}^{10}\operatorname{AP}_{{\rm IF},r}\right].
\end{equation}
The project reports average precision (AP) as AUPRC; AP weights precision by recall increments, $\sum_t(R_t-R_{t-1})P_t$. It summarizes ranking across thresholds and is not classification accuracy. Isolation Forest is evaluated over ten seeds with 500 trees per seed, as specified in the original project protocol.

\textbf{Objectives and scope.} We seek to (i) improve both detectors through valid normal-data generation, (ii) support both mixed-attribute datasets through a common orchestration interface, and (iii) identify which transformations transfer across datasets. The implementation uses different generator and selector configurations; shared orchestration does not imply identical statistical estimators.

\textbf{Contributions.} We present empirical characterization of normal and attack regimes, two technically explicit generation configurations, a reproducible implementation, and an experimental analysis including unsuccessful approaches. The principal finding is that normal-distribution fidelity is insufficient as a standalone objective: useful transformations depend on both the data regime and the consuming detector.
''')

eda = between(r'\section{Data Findings and Exploratory Analysis}\label{sec:eda}', '% ---------------- 3 Methodology')
eda = re.sub(r'\\begin\{figure\}.*?\\end\{figure\}', '', eda, flags=re.S)
eda = eda.replace('All numbers in this section are recomputed from', 'The supplied report derives these statistics from')
eda = eda.replace('so they must not be neutralized', 'so the implementation protects these sparse columns from neutralization')
eda = eda.replace('so valid normal data actively confuses the detectors', 'indicating weak utility from directly resampling the normal distribution')
eda = eda.replace('contain essentially no duplicate rows', 'contain few exact duplicate rows')
eda = eda.replace('a flat ECDF', 'a flatter density and approximately linear ECDF')
eda = eda.replace('so one global distribution is the wrong model', 'motivating traffic-regime conditioning')
eda = eda.replace('a Campus-LAN segment', 'a segment interpreted as Campus-LAN traffic')
eda = eda.replace('and must not', 'and should not')
eda = re.sub(r'\\paragraph\{UNSW-NB15:.*?(?=\\paragraph\{NSL-KDD:)', r'''\\paragraph{UNSW-NB15: heterogeneous traffic regimes.} The bootstrap baseline is 0.4210, motivating changes to normal support. The supplied EDA identifies 118 protocol/service/state cells: attack shares range from 0.8\\% in udp/dns/CON to 99\\% in udp/dns/INT. Source TTL values 31 and 32 contain 64.8\\% of normal rows and no observed attacks, indicating a dominant normal regime. Single-feature ECOD is useful for a small subset of TTL and connection-count features but weak for several byte, packet, and loss measurements (Figure~\\ref{fig:edau}). These are observed training relationships, not guarantees about hidden attacks.

''', eda, flags=re.S)
eda = re.sub(r'\\paragraph\{NSL-KDD:.*?(?=\\paragraph\{Engineering implications)', r'''\\paragraph{NSL-KDD: strong empirical normal support.} The bootstrap baseline is 0.9292. Attacks concentrate in particular connection types: 98.3\\% of attacks have \\texttt{logged\\_in}=0, flag S0 is 98\\% attack, and ICMP is 75\\% attack. Several numerical features have weak independent ECOD utility, while sparse content indicators may still expose unseen attacks and are protected from neutralization (Figure~\\ref{fig:edan}).

''', eda, flags=re.S)
data_table = re.search(r'\\begin\{table\}.*?\\end\{table\}', eda, re.S).group()
eda = r'''
Both supplied datasets contain mixed numerical and categorical network records. NSL-KDD~\cite{nsl} describes connections through counts, rates, protocol/service fields, and content indicators. UNSW-NB15~\cite{unsw} describes flows through packet statistics, timings, TTLs, and connection state. Table~\ref{tab:data} gives the project subsets, rather than the full benchmark sizes.
''' + data_table + r'''
\paragraph{UNSW-NB15: heterogeneous traffic regimes.}
The bootstrap baseline is 0.4210. Across 118 protocol/service/state cells, observed attack shares range from 0.8\% in udp/dns/CON to 99\% in udp/dns/INT. Source TTLs 31 and 32 contain 64.8\% of normals and no observed attacks. Figure~\ref{fig:edau} supports conditional generation and selective treatment of weak features; these training relationships do not guarantee separation of unseen attacks.

\paragraph{NSL-KDD: strong empirical normal support.}
The baseline is 0.9292; 98.3\% of observed attacks have \texttt{logged\_in}=0, and flag S0 is 98\% attack. Several numerical columns have weak independent utility, while sparse content indicators may expose unseen attacks (Figure~\ref{fig:edan}). The design therefore retains normal modes, adds local variation, and protects sparse content features.

\textbf{Engineering implication.} UNSW motivates conditional redistribution; NSL-KDD motivates preservation. Both require explicit validity rules and evaluation of detector utility beyond marginal fidelity.
'''
pages.append(r'\section{Dataset Characteristics and Exploratory Analysis}\label{sec:eda}' + eda +
figure('eda_unsw', 'UNSW training diagnostics: attack share by traffic cell, source TTL, and single-feature ECOD utility; dashed utility line denotes attack prevalence.', 'fig:edau', r'0.94\linewidth') +
figure('eda_nsl', 'NSL-KDD training diagnostics: connection regimes, service-count overlap, and single-feature ECOD utility.', 'fig:edan', r'0.94\linewidth'))

pages.append(r'''
\section{Methodology}\label{sec:method}
\subsection{Shared Processing Architecture}
The pipeline separates statistical generation from common post-processing. Labeled inputs support normal extraction and utility estimation; attack rows inform selected transformations but are never exported as normal training records. Both configurations pass through the same functions, with neutral settings disabling irrelevant transformations.
\begin{figure}[H]\centering
\begin{tikzpicture}[font=\small,node distance=3mm,
st/.style={draw=navy,fill=navy!6,rounded corners=2pt,minimum height=9mm,minimum width=19mm,align=center,inner sep=2pt}]
\node[st](a){Load +\\profile};\node[st,right=of a](b){Generate};
\node[st,right=of b](c){Weight};\node[st,right=of c](d){Jitter +\\stretch};
\node[st,right=of d](e){Constrain};\node[st,right=of e](f){Neutralize};
\node[st,right=of f](g){Cap +\\export};
\foreach\x/\y in {a/b,b/c,c/d,d/e,e/f,f/g}{\draw[-{Stealth[length=1.5mm]}](\x)--(\y);}
\end{tikzpicture}
\caption{Shared orchestration. NSL-KDD disables category weighting, duplicate jitter, and the separate tail-stretch stage. Local copula tail slack remains active.}\label{fig:pipe}
\end{figure}
\subsection{Common Statistical and Engineering Principles}
\textbf{Mixed attributes.} Categorical combinations are retained jointly rather than independently recombined. Numeric counts, bounded rates, binary indicators, and continuous measurements receive different treatment, preventing a continuous model from inventing fractional indicators or impossible protocol combinations.

\textbf{Conditional Gaussian copulas.} For a group $g$, each numerical variable is transformed through its empirical distribution and the standard normal inverse CDF:
\begin{equation}
u_{ij}=\widehat F_{gj}(x_{ij}),\quad z_{ij}=\Phi^{-1}(u_{ij}),\quad
\widetilde z\sim\mathcal N(0,R_g),\quad
\widetilde x_j=\widehat F_{gj}^{-1}\!\left(\Phi(\widetilde z_j)\right).
\end{equation}
This separates the shape of each marginal from dependence in Gaussian rank space. Correlation matrices are regularized to be positive definite; small groups borrow global correlation estimates. The model approximates dependence and does not guarantee preservation of every nonlinear relationship.

\textbf{Feature neutralization.} A selected column is replaced independently across output rows by $U[Q_{0.01},Q_{0.99}]$, using quantiles of real normal observations; integer-valued columns are rounded. These bounds describe the central 98\% probability interval, not a mean $\pm3\sigma$ range. Uniform density produces an approximately linear ECDF and changes tail penalties; it does not make ECOD scores constant. Neutralization also deliberately breaks dependencies involving the replaced column.

\begin{table}[H]\centering\small
\begin{tabular}{P{2.6cm}P{6.2cm}P{6.2cm}}\toprule
Component & UNSW-NB15 configuration & NSL-KDD configuration\\\midrule
Normal foundation & 35\,000 labeled normal rows; contrastive weighted resampling & 19\,600 normal rows after 2\% Isolation Forest scrub\\
Grouping & Joint protocol/service/state conditioning & SILK-style seeding and k-FreqItems refinement\\
Generation & 70\,000 DVS-copula rows; five sampling bags & 20\,400 local-copula rows plus all foundation rows\\
Main parameters & $\alpha=35$, $\tau_u=0.15$, $\tau_{low}=0.05$ & Historical default $k=20$, $\rho=0.35$; later variant $18$, $0.30$\\
Resampling / tails & Weight scale 3; repeated-row jitter $0.10\sigma$; gated stretch 10 & Weight scale 0; jitter off; separate stretch factor 1\\
Neutralization & Four empirically selected columns & Six columns; sparse-content guard\\
Output & 70\,000 rows; schema and numeric checks & 40\,000 rows; schema and numeric checks\\\bottomrule
\end{tabular}
\caption{Configuration of the historical scored pipelines, with later NSL-KDD tuning distinguished explicitly.}\label{tab:cfg}
\end{table}
\textbf{Design rationale.} The same interface exposes where the datasets differ: UNSW-NB15 requires substantial reweighting and dispersion, whereas NSL-KDD retains a strong empirical foundation. The following two subsections use the same five-part structure: preparation, grouping, generation, post-processing, and rationale.
''')

pages.append(r'''
\subsection{UNSW-NB15: Contrastive Conditional Copula Generation}
\subsubsection{Input Preparation and Feature Assessment}
The generator separates 35\,000 normal rows from 15\,000 attack rows and identifies categorical, integer-valued, and continuous columns. For each numerical feature $j$, ECOD is fitted on normal values and scores all labeled rows. Its utility is $c_j=\operatorname{Corr}(s_j^{\rm ECOD},y)$; the high-utility set is $H=\{j:c_j\ge0.15\}$. This is an in-sample selection signal, not an independent estimate of generalization.
\subsubsection{Grouping and Normal-Support Selection}
Contrastive purification favors normal observations whose useful feature values occur more often among normals than among attacks:
\begin{equation}
\ell_i=\sum_{j\in H}\log\frac{\widehat p_{0j}(x_{ij})+10^{-6}}{\widehat p_{1j}(x_{ij})+10^{-6}},\qquad
w_i=\frac{\exp(\operatorname{clip}(\ell_i,-10,10))}{\sum_l\exp(\operatorname{clip}(\ell_l,-10,10))}.
\end{equation}
The implementation uses empirical exact-value frequencies, with $10^{-4}$ for unseen values; it does not fit continuous kernel densities. It resamples 35\,000 normals with replacement using $w_i$. This can suppress legitimate rare regimes, so purification is a utility-driven trade-off rather than verified removal of mislabeled records.

The historical final generator groups purified rows by joint \texttt{proto|service|state}. For a group of size $m_g$, $R_g^*=\lambda_gR_g+(1-\lambda_g)R_0$ with $\lambda_g=m_g/(m_g+100)$; groups below 50 rows use the global correlation. Jointly resampling categorical tuples avoids inventing unsupported combinations.
\subsubsection{Synthetic Sample Generation and Allocation}
Copula sampling reconstructs numerical marginals inside each regime. Discriminative variance scaling then adds independent noise to eligible low-utility columns:
\begin{equation}
\widetilde x_{ij}\leftarrow\widetilde x_{ij}+\epsilon_{ij},\qquad
\epsilon_{ij}\sim\mathcal N\!\left(0,[35(0.05-c_j)_+\sigma_j]^2\right).
\end{equation}
Protected discrete indicators are excluded; other integer columns may be perturbed and then rounded. Expanded values are clipped from the observed normal minimum to $1.5$ times the positive normal maximum; high-utility and protected features retain observed bounds. Five sampling bags produce 70\,000 rows. Independent dispersion can weaken copula dependencies, so dependence is modeled before perturbation rather than guaranteed afterward.
\subsubsection{Post-processing and Domain Validation}
Category weights resample the pool with probability proportional to $\exp(3\theta_c)$ across 14 fitted cells. Duplicate copies receive $0.10\sigma_j$ Gaussian jitter on eligible continuous features. A separate tail-stretch rule uses four attack clusters: a direction must meet AUC $0.58$ and agreement in at least 75\% of clusters; the opposite normal tail is stretched tenfold about its 10th or 90th percentile. For example, if attacks are consistently low, an upper-tail value becomes $x'=Q_{0.90}+10(x-Q_{0.90})$.

Domain rules enforce non-negativity, integer counts, TTL bounds, protocol-specific zeroing, and TCP timing consistency. Uniform filling replaces \texttt{sinpkt}, \texttt{sjit}, \texttt{ct\_dst\_src\_ltm}, and \texttt{ct\_dst\_ltm}; final caps constrain selected counts to 100 and \texttt{dmean} to 1500. These operations address different issues: constraints repair invalid values, while neutralization changes detector sensitivity.
\subsubsection{Rationale and Expected Detector Effect}
Conditioning avoids averaging unrelated traffic regimes; purification changes which normal modes dominate training; DVS widens weak dimensions; tail stretching changes one-sided support. Their benefits are empirical. A normal row becoming less influential or a marginal becoming broader does not establish that it was erroneous or that unseen attacks will be separated more effectively.
''')

pages.append(r'''
\subsection{NSL-KDD: Mixed-Attribute Clustering and Local Copula Generation}
\subsubsection{Input Preparation and Feature Assessment}
Isolation Forest removes the most readily isolated 2\% of 20\,000 normal observations, leaving the 19\,600-row EDA foundation. The retained observations are preserved through the generation stage. Later constraints and neutralization can still modify their values. Binary and low-cardinality integer features remain exact; continuous variables are discretized into up to five quantile bins, with duplicate quantile edges collapsed.
\subsubsection{Grouping and Normal-Support Selection}
A row becomes an itemset such as $\{\texttt{protocol=tcp},\texttt{service=http},\texttt{src\_bytes=q3},\ldots\}$. An item is a feature--value token, not an entire record. Five bins provide a coarse ordinal representation while avoiding excessive sparsity; this is a design setting rather than a proven optimum.

MinHash~\cite{minhash} maps each set to 48 minimum-hash values using deterministic FNV-style token hashing and seeded affine modular transforms. LSH splits this signature into 12 bands of four values and collects identical bands into candidate buckets. Hashing proposes seeds only: subsequent clustering uses original token sets, not distances between hash integers. SILK-style seeding constructs frequent-token candidate centres and selects separated candidates with probability proportional to bucket weight times squared distance, adapting k-means++~\cite{kmpp} to Jaccard distance:
\begin{equation}
d_J(A,B)=1-\frac{|A\cap B|}{|A\cup B|},\qquad
C_j=\left\{t:\frac{\#\{i\in I_j:t\in R_i\}}{|I_j|}\ge\rho\right\}.
\end{equation}
K-FreqItems alternates nearest-centre assignment and frequency-centre updates until assignments stabilize or 25 iterations are reached. The historical default uses $k=20$, $\rho=0.35$; the later $k=18$, $\rho=0.30$ variant uses the same procedure. Here $\rho$ is within-cluster item frequency, not association-rule confidence. Clusters below 500 rows merge into a nearest sufficiently large cluster; the final number can therefore be below $k$. This is a SILK-inspired implementation, not a claim of reproducing every step of a published SILK algorithm.
\subsubsection{Synthetic Sample Generation and Allocation}
For cluster $j$ with $n_j$ foundation rows, its additional-row budget is
\begin{equation}
a_j=20\,400\frac{n_j}{19\,600}.
\end{equation}
Floors are assigned first; remaining rows go to the largest fractional remainders, with deterministic label tie-breaking. Each sufficiently large cluster has two seeded, protocol-conditioned copula models; global models provide a fallback. A tail-slack factor of 1.05 mildly expands eligible continuous upper tails while bounded rates remain protected. Thus 19\,600 foundation rows plus 20\,400 generated rows produce exactly 40\,000 records before post-processing.
\subsubsection{Post-processing and Domain Validation}
Separate category weighting, duplicate jitter, and gated tail stretching are disabled. Constraints enforce rates in $[0,1]$, non-negative integer counts, and valid indicators. Six columns receive uniform fill: \texttt{srv\_count}, \texttt{srv\_diff\_host\_rate}, \texttt{duration}, \texttt{dst\_host\_count}, \texttt{dst\_host\_same\_src\_port\_rate}, and \texttt{dst\_host\_rerror\_rate}. Selection uses single-feature AP lift above attack prevalence of at most 0.10, protects columns with a dominant value share of at least 0.90, and excludes \texttt{dst\_host\_srv\_diff\_host\_rate} after an adverse experiment.
\subsubsection{Rationale and Expected Detector Effect}
The real-normal baseline is already strong, so proportional local synthesis adds variation without deliberately equalizing rare and common modes. Itemsets accommodate categories and discretized measurements in one similarity measure. This avoids Euclidean coding assumptions but does not eliminate high-dimensional similarity concentration or prove an optimal partition.
''')

pages.append(r'''
\section{Implementation and Reproducibility}\label{sec:impl}
\subsection{Code Organization and Artifact Flow}
\texttt{final\_pipeline/pipeline.py} loads configuration, dispatches a generator, applies transformations, and writes ordered CSV columns with unique IDs. \texttt{config.py} stores target sizes, seeds, generator choices, selector settings, and artifact paths. The vendor modules contain DVS, SILK-style clustering, the hybrid allocator, copula sampling, and domain constraints. \texttt{expt-17/src/run\_expt17.py} exposes NSL cluster parameters and UNSW condition-mixture variants; Expt-19 contains the later UNSW conditioning experiments.

The NSL foundation CSV, category weights, and feature-selection tables are persisted artifacts. Their reuse makes a run reproducible but means the export command is not a fresh end-to-end fit of every upstream decision. Results must be associated with a configuration and data snapshot; changing defaults can produce a different file even when the command is unchanged.
\begin{figure}[H]\centering
\fbox{\begin{minipage}{0.94\linewidth}\small
\textbf{Algorithm 1: Configurable normal-data generation and export}\\[-2pt]
\rule{\linewidth}{0.4pt}
\textbf{Input:} labeled data $D$, configuration $\theta$, saved artifacts $A$, target $m$, seed $s$.\\
\textbf{Output:} normal-only table $S$ with $m$ records and ordered feature columns.
\begin{enumerate}[label=\arabic*.,leftmargin=2em,itemsep=3pt]
\item Load $D$; separate labels, categorical columns, and numerical columns.
\item \textbf{If} the generator is contrastive DVS:
  estimate feature utility; reweight normals; fit conditional copulas; sample $m$ rows; apply selective dispersion.
\item \textbf{Otherwise, if} the generator is the SILK hybrid:
  load the scrubbed foundation; tokenize; form MinHash--LSH candidates; choose seeds; refine k-FreqItems clusters; allocate $m-|D_0|$ rows proportionally; sample local copulas; combine with $D_0$.
\item Apply configured category-cell weights by sampling with replacement.
\item Jitter repeated rows on eligible continuous columns when enabled.
\item Estimate consistent attack directions and stretch opposite normal tails when enabled.
\item Enforce dataset domain rules; replace selected weak columns using normal quantile bounds; apply final hard caps.
\item Restore feature order; add unique IDs; assert row count, no missing values, and unique IDs; export CSV.
\item Compare against a reference artifact when available; record configuration and diagnostics for interpretation.
\end{enumerate}
\end{minipage}}
\caption{Implementation-level pseudocode matching the shared transformation order. Neutral parameters disable optional stages; reference comparison is a separate verification operation.}\label{fig:algorithm}
\end{figure}
\subsection{Computational Characteristics and Validation Boundaries}
With $n$ rows, $d$ tokens per row, and $h=48$ hashes, signature construction costs $O(ndh)$. Direct Jaccard assignment is approximately $O(Tnkd)$ for $T\le25$ iterations and bounded-size centres. LSH reduces seed-search work but does not replace these full assignment passes. Copula covariance estimation scales as $O(nd^2)$ and matrix factorization as $O(d^3)$ per fitted group; small-group regularization is important for numerical stability.

The implementation uses pandas/NumPy, SciPy, scikit-learn, and PyOD in an in-memory Python workflow. No measured distributed speedup, throughput, or latency result is available, so scalability claims are limited to these algorithmic costs. The provided table sizes are moderate; larger deployments would require profiling, batched signatures, and group-parallel fitting.

\textbf{Audit scope.} Export assertions cover row count, unique IDs, and missing values; they do not explicitly reject infinity or establish statistical fidelity. Finite-value and cross-feature domain checks are additional recommended release checks, especially because neutralization follows constraints. Reference equality within tolerance verifies reproducibility of values, not generalization to unseen attacks.
''')

def table(label):
    for match in re.finditer(r'\\begin\{table\}.*?\\end\{table\}', source, re.S):
        if rf'\label{{{label}}}' in match.group(): return match.group().replace('[t]', '[H]').replace('[h]', '[H]').replace('[!t]', '[H]')
    raise ValueError(label)
pages.append(r'''
\section{Experimental Design and Comparative Results}\label{sec:exp}
\subsection{Evaluation Protocol and Score Provenance}
Historical public submissions provide the ablations below. A 14-fold local harness simulated distribution shifts through held-out attack clusters, categorical groups, and tail regions; it screened candidates but did not reliably reproduce leaderboard rankings. Each reported score belongs to a particular evaluation partition. Later hidden scores must not be subtracted from historical public scores to infer an improvement.

\subsection{Cumulative Improvements on Both Datasets}
'''+ figure('journey', 'Historical public-score progression for the two datasets. These are method comparisons, not precision--recall curves.', 'fig:journey', r'0.94\linewidth')+
table('tab:uns').replace('single DVS on joint proto$|$service$|$state (final)', 'single DVS on joint proto$|$service$|$state')+
table('tab:nsl').replace('The last step is within noise.', 'The small final increment has no reported confidence interval.')+r'''
\textbf{Interpretation.} UNSW gains 0.19352 over its historical baseline, primarily from contrastive DVS. NSL-KDD gains approximately 0.00814, mainly through feature neutralization. Cumulative improvements do not isolate each component's causal contribution.
''')

neg = table('tab:neg').replace('P{3.8cm}P{1.6cm}P{1.5cm}P{8.2cm}', 'P{3.3cm}P{1.3cm}P{1.4cm}P{8.2cm}')
neg = neg.replace('Why it failed', 'Interpretation / plausible mechanism')
neg = neg.replace('Optimizes fidelity; sparse categories and detector-relevant tails are not preserved; UNSW is worse than real normals', 'Adversarial training did not improve detector utility; insufficient tail or category coverage is a plausible explanation, not a measured convergence failure')
neg = neg.replace('Hard rejection over-contracts the normal support and removes useful normal modes', 'Consistent with excessive support contraction; removal of useful modes was not isolated by a dedicated causal ablation')
neg = neg.replace('Protocol structure is under-represented', 'Increasing state-conditioned share lowered the score; reduced protocol-specific coverage is a hypothesis')
pages.append(r'''
\subsection{Sensitivity Analysis and Unsuccessful Alternatives}
Negative experiments delimit the useful configuration space. Table~\ref{tab:neg} retains historical scores while distinguishing observed regressions from plausible explanations. We did not measure GAN mode collapse, vanishing gradients, or convergence failures, and therefore do not attribute the CTGAN result to those mechanisms.
'''+neg+figure('sensitivity', 'UNSW sensitivity: historical leaderboard performance by weight scale and local-harness changes for alternative fill rules. These panels use different evaluation sources and should be interpreted separately.', 'fig:sens')+r'''
\textbf{Implication.} More balancing, stronger dispersion, or finer conditioning can change useful normal prevalence as well as coverage. Their effect depends on the existing generator and post-processing configuration. A failed variant rejects that tested configuration; it does not establish that the underlying model family is universally unsuitable.
''')

pages.append(r'''
\subsection{Final Scores and Evaluation Provenance}
Table~\ref{tab:final} reports the selected configurations and their final hidden scores: 0.68508 for UNSW-NB15 and 0.95247 for NSL-KDD. These results use the supplied report's final evaluation record. Later parameter variants are listed separately; they were not the historically selected submissions. The final hidden and historical public partitions differ, so their absolute scores are not directly comparable.
\begin{table}[H]\centering\small
\begin{tabular}{P{3.0cm}P{7.2cm}P{2.4cm}P{1.6cm}}\toprule
Dataset / record & Configuration or source & Evaluation record & AUPRC\\\midrule
UNSW selected & Joint proto/service/state DVS; supplied report & Final hidden & 0.68508\\
NSL selected & SILK $k=20$, frequency 0.35; supplied report & Final hidden & 0.95247\\
UNSW later variant & \texttt{submission\_unsw\_v2\_state35.csv} & Later reported best & 0.68651\\
NSL later variant & \texttt{submission\_nslkdd\_v2\_k18\_cf30.csv} & Later reported best & 0.95276\\\bottomrule
\end{tabular}
\caption{Score provenance. Later variants are distinct from the selected historical configurations; no matched-split causal comparison is asserted.}\label{tab:final}
\end{table}
\subsection{Local Validation Reliability and Residual Errors}
'''+table('tab:har').replace('Disagreements on weights and NSL fill made the leaderboard the arbiter.', 'Disagreements expose distribution-shift sensitivity and the risk of repeated leaderboard tuning.')+
figure('clusters', 'UNSW in-sample attack-cluster AUROC, with detectors trained on baseline versus generated normals. This diagnostic uses AUROC rather than AUPRC and is not an independent hidden-test estimate.', 'fig:clu', r'0.50\linewidth')+r'''
The generated UNSW distribution improves the large TCP cluster from AUROC 0.496 to 0.775 but lowers the UDP cluster from 0.913 to 0.827. Aggregate gains can therefore conceal regressions on individual attack regimes. A supervised diagnostic performs substantially better in the difficult TCP cell, suggesting that joint discriminative structure remains underused by the fixed detectors; this is not a proof of an unsupervised performance ceiling.

Only scalar hidden scores are available in the supplied evidence. A genuine hidden precision--recall curve requires per-row labels and anomaly scores and cannot be reconstructed from AUPRC alone. Accordingly, this report uses score comparisons and explicitly labeled local diagnostics rather than fabricated hidden curves.
''')

pages.append(r'''
\section{Lessons Learned and Research Limitations}\label{sec:lessons}
\paragraph{1. Downstream utility and distribution fidelity are distinct objectives.}
CTGAN achieved 0.3925 on UNSW compared with 0.4210 for bootstrap normals, whereas the historical DVS pipeline reached 0.61452. Uniform filling also improved over constant filling. These comparisons favor detector-aware selection in this task, but do not demonstrate that unrealistic data are generally desirable. Validity, support coverage, and detection utility must be assessed separately.

\paragraph{2. Dataset structure determines which transformations are useful.}
NSL-KDD's strong real-normal baseline favors preservation and modest local synthesis. UNSW's heterogeneous traffic cells support conditional generation and more extensive redistribution. Transferring aggressive purification to NSL-KDD reduced its score to approximately 0.899. A shared software interface is valuable, but selecting different generators is a substantive methodological difference that must be disclosed.

\paragraph{3. Local models require sufficient support and regularization.}
Conditioning improved UNSW incrementally, while small-cluster merging and correlation shrinkage made copula fitting more stable. Excessively fine partitions or more mixture components did not consistently improve results. Group size, fallback behavior, and feature dependence matter more than the number of clusters alone. K-FreqItems frequency thresholds describe cluster prototypes; they do not certify statistical significance.

\paragraph{4. Proxy validation can mis-rank apparently promising changes.}
Forty-cell weights improved the local held-out proxy by 0.0069 but reduced the public score by 0.0206. Such reversals demonstrate that a shift simulation is incomplete. Repeated public-score tuning also creates selection bias. Future experiments should freeze a validation protocol, regenerate every learned artifact within each training fold, and reserve an untouched evaluation set.

\paragraph{5. Small gains require uncertainty estimates and matched comparisons.}
The final historical SILK increment was approximately 0.0006. No repeated hidden evaluation or confidence interval establishes its statistical significance; it should be described as a small observed gain. Bootstrap uncertainty on a fixed test set, multiple generation seeds, and paired detector comparisons would strengthen claims. Scores from different hidden/public partitions cannot establish generalization gaps.

\paragraph{6. Causal explanations require targeted ablations.}
Tail shaping, purification, and uniform fill plausibly alter false-positive behavior, but their mechanisms were not fully isolated. Additive noise and independent filling can also break learned dependence. Future work should measure per-regime false-positive rates, attack recall, and coverage before and after each transformation, rather than infer mechanisms solely from aggregate AUPRC.

\section{Conclusion and Future Work}
The project demonstrates a configurable workflow for generating valid normal-only training tables for fixed anomaly detectors. Historical public improvements were substantial on UNSW-NB15 and smaller on NSL-KDD, reflecting different starting distributions. Conditional contrastive copulas and selective dispersion served UNSW; preserving normal observations and generating within mixed-attribute clusters served NSL-KDD. The evidence supports empirical configuration selection, not a universal superiority claim for clustering or DVS.

Future work should use nested selection for weights and feature filling, measure uncertainty, and test restoration of suppressed normal regimes. Computational profiling and batched or parallel group fitting would be needed before making large-scale deployment claims.

\paragraph{Reproducibility and authorship.}
The shared entry point is \texttt{python final\_pipeline/pipeline.py unsw} or \texttt{nslkdd}; \texttt{verify.py} compares exports with available scored references. Exact reproduction also requires matching configuration, artifacts, and dependencies. Existing figures are reused from the original report's \texttt{figs.py} outputs; this revision did not rerun training or hidden evaluation. The authors retain responsibility for methods and results. The original report disclosed Claude assistance; this separate revision additionally used OpenAI Codex for technical editing, organization, and document preparation.
''')

refs = r'\begin{thebibliography}' + source.split(r'\begin{thebibliography}',1)[1]
refs = refs.replace(r'\emph{IEEE TKDE}, 2022.', r'\emph{IEEE TKDE}, 2022. \url{https://doi.org/10.1109/TKDE.2022.3159580}.')
refs = refs.replace(r'\emph{NeurIPS}, 2019.', r'\emph{NeurIPS}, 2019. \url{https://arxiv.org/abs/1907.00503}.')
text = preamble + r'\hypersetup{pageanchor=false}\begin{titlepage}' + cover + r'\end{titlepage}\setcounter{page}{1}\hypersetup{pageanchor=true}'
text += '\n\\clearpage\n'.join(pages)
text += '\n\\clearpage\n' + refs
text = text.replace('[h]', '[H]').replace('[t]', '[H]')
text = text.replace('p{3.7cm}', 'p{2.8cm}')
text = text.replace(r'\_', r'\_\allowbreak ')
text = text.replace('0.922/0.929', r'0.922/\allowbreak 0.929').replace('0.936/0.937', r'0.936/\allowbreak 0.937')
out = ROOT / 'revised'
out.mkdir(exist_ok=True)
(out / 'report_revised.tex').write_text(text)
print(out / 'report_revised.tex')
