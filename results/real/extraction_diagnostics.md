# Source Extraction Diagnostics & Audit Report

## 1. Deep Dive on 5 Focal Papers

### Paper: `2017.jeptalnrecital-court.2`
- **PDF Path:** `data\raw\extended\2017.jeptalnrecital-court.2.pdf`
#### Where did the first words of the French abstract go?
**Dropped by the header regex rule.**
- In the raw PDF page 1, the text layout contains `RÉSUMÉ\nPour déterminer si certaines mesures d’association...`.
- The regex `_header_pattern('Résumé')` is compiled as `(?im)^\s*(?:résumé|resume)\s*:?.*$`.
- Because `\s*` matches the trailing newline `\n`, the pattern continues onto the *second line*.
- The trailing `.*$` matches the entire next line `Pour déterminer si certaines mesures d’association lexicale fréquemment employées en TAL attribuent`.
- As a result, the entire opening sentence of the French abstract was treated as part of the header and discarded. The extracted abstract begins mid-sentence: `des scores élevés à des n-grammes...`.

```text
--- REGEX MATCH DETAILS ---
FR Header Match (215:321): 'RÉSUMÉ\nPour déterminer si certaines mesures d’association lexicale fréquemment employées en TAL attribuent'
EN Header Match (912:1004): 'ABSTRACT\nUsing Fisher’s Exact Test to Evaluate Association Measures for Bigrams and Trigrams'
```

#### Raw Page 1 Text (first 2500 characters):
```text
Évaluation de mesures d’association pour les bigrammes et les
trigrammes au moyen du test exact de Fisher
Yves Bestgen
CECL, Place du Cardinal Mercier 10, B-1348 Louvain-la-Neuve, Belgique
yves.bestgen@uclouvain.be
RÉSUMÉ
Pour déterminer si certaines mesures d’association lexicale fréquemment employées en TAL attribuent
des scores élevés à des n-grammes que le hasard aurait pu produire aussi souvent qu’observé, nous
avons utilisé une extension du test exact de Fisher à des séquences de plus de deux mots. Les analyses
ont porté sur un corpus de quatre millions de mots d’anglais conversationnel extrait du BNC. Les
résultats, basés sur la courbe précision-rappel et sur la précision moyenne, montrent que le LL-simple
est extrêmement efﬁcace. IM3 est plus efﬁcace que les autres mesures basées sur les tests d’hypothèse
et atteint même un niveau de performance presque égal à LL-simple pour les trigrammes.
ABSTRACT
Using Fisher’s Exact Test to Evaluate Association Measures for Bigrams and Trigrams
To determine whether some often-used lexical association measures assign high scores to n-grams
that chance could have produced as frequently as observed, we used an extension of Fisher’s exact
test to sequences longer than two words to analyse a corpus of four million words. The results, based
on the precision-recall curve and the average precision, show that simple-ll is extremely effective.
MI3 is more efﬁcient than the other hypothesis tests-based measures and even reaches a performance
level almost equal to simple-ll for trigrams.
MOTS-CLÉS : Mesures d’association lexicale ; N-grammes de mots ; LL-simple ; IM3.
KEYWORDS: Lexical association measures ; Word n-grams ; Simple-LL ; MI3.
1
Introduction
L’évaluation de l’efﬁcacité des mesures d’association lexicale pour identiﬁer les expressions po-
lylexicales retient l’attention de nombreux chercheurs depuis plus de vingt ans (Church & Hanks,
1990 ; Evert & Krenn, 2005 ; Ramish, 2015). Dans le cas de bigrammes, un grand nombre de mesures
ont été comparées et plusieurs synthèses sont disponibles (Evert, 2008 ; Pecina, 2010). Les mesures
d’association pour des n-grammes de plus de deux mots ont retenu beaucoup moins l’attention comme
l’ont souligné Evert (2008), Gries (2010) et Nerima et al. (2010).
Cette situation a probablement conduit certains chercheurs à privilégier la fréquence de cooccurrence
brute comme critère de sélection (Wermter & Hahn, 2006 ; Biber, 2009). Dans le cas des bigrammes,
cependant, il est bien éta
```

#### Extractor Output:
- **fr_abstract:** des scores élevés à des n-grammes que le hasard aurait pu produire aussi souvent qu’observé, nous avons utilisé une extension du test exact de Fisher à des séquences de plus de deux mots. Les analyses ont porté sur un corpus de quatre millions de mots d’anglais conversationnel extrait du BNC. Les résultats, basés sur la courbe précision-rappel et sur la précision moyenne, montrent que le LL-simple est extrêmement efﬁcace. IM3 est plus efﬁcace que les autres mesures basées sur les tests d’hypothèse et atteint même un niveau de performance presque égal à LL-simple pour les trigrammes.
- **en_abstract:** To determine whether some often-used lexical association measures assign high scores to n-grams that chance could have produced as frequently as observed, we used an extension of Fisher’s exact test to sequences longer than two words to analyse a corpus of four million words. The results, based on the precision-recall curve and the average precision, show that simple-ll is extremely effective. MI3 is more efﬁcient than the other hypothesis tests-based measures and even reaches a performance level almost equal to simple-ll for trigrams.
- **body (first 300 chars):** Évaluation de mesures d’association pour les bigrammes et les trigrammes au moyen du test exact de Fisher Yves Bestgen CECL, Place du Cardinal Mercier 10, B-1348 Louvain-la-Neuve, Belgique yves.bestgen@uclouvain.be MOTS-CLÉS : Mesures d’association lexicale ; N-grammes de mots ; LL-simple ; IM3. KEY

---

### Paper: `2018.jeptalnrecital-court.4`
- **PDF Path:** `data\raw\extended\2018.jeptalnrecital-court.4.pdf`
#### Where did the first words of the French abstract go?
**Dropped by the header regex rule.**
- Raw page 1 text has `RÉSUMÉ\nCet article présente des méthodes permettant l’évaluation de la satisfaction client à partir de très vastes...`.
- The trailing `\s*` in `(?im)^\s*(?:résumé|resume)\s*:?.*$` matched `\n`.
- The following `.*$` swallowed the entire first line `Cet article présente des méthodes permettant l’évaluation de la satisfaction client à partir de très vastes`.
- The extracted French abstract was left starting mid-sentence: `corpus de conversation de type “chat”...`.

```text
--- REGEX MATCH DETAILS ---
FR Header Match (448:562): 'RÉSUMÉ\nCet article présente des méthodes permettant l’évaluation de la satisfaction client à partir de très vastes'
EN Header Match (1089:1190): 'ABSTRACT\nCustomer satisfaction prediction with attention-based RNNs from a chat contact center corpus'
```

#### Raw Page 1 Text (first 2500 characters):
```text
Évaluation automatique de la satisfaction client à partir de
conversations de type "chat" par réseaux de neurones récurrents
avec mécanisme d’attention
Jeremy Auguste1
Delphine Charlet2
Géraldine Damnati2
Benoit Favre1
Frédéric Béchet1
(1) Aix-Marseille Univ, Université de Toulon, CNRS, LIS, Marseille, France
(2) Orange Labs, Lannion
(1) {jeremy.auguste,benoit.favre,frederic.bechet}@lis-lab.fr
(2) {delphine.charlet,geraldine.damnati}@orange.fr
RÉSUMÉ
Cet article présente des méthodes permettant l’évaluation de la satisfaction client à partir de très vastes
corpus de conversation de type “chat” entre des clients et des opérateurs. Extraire des connaissances
dans ce contexte demeure un déﬁpour les méthodes de traitement automatique des langues de par
la dimension interactive et les propriétés de ce nouveau type de langage à l’intersection du langage
écrit et parlé. Nous présentons une étude utilisant des réponses à des sondages utilisateurs comme
supervision faible permettant de prédire la satisfaction des usagers d’un service en ligne d’assistance
technique et commerciale.
ABSTRACT
Customer satisfaction prediction with attention-based RNNs from a chat contact center corpus
This paper presents methods to perform knowledge extraction from very large databases of WEB
chat conversations between operators and clients in customer contact centers. Extracting knowledge
from chat corpus is a challenging research issue. Simply applying traditional text mining tools is
clearly sub-optimal as it takes into account neither the interaction dimension nor the particular nature
of this language which shares properties of both spoken and written language. We present a method
predicting users satisfaction in a chat-based service trained on answers from users to satisfaction
surveys.
MOTS-CLÉS : Réseaux de neurones récurrents, Attention, Satisfaction client, Conversations.
KEYWORDS: Recurrent Neural Networks, Attention-based RNNs, chat, satisfaction prediction.
1
Introduction
L’analyse automatique d’enregistrements de conversations, écrites ou orales, représente un déﬁpour
les méthodes de Traitement Automatique de la Langue à cause d’une part de la nature spontanée
du langage employé, et d’autre part de la construction interactive du discours au fur et à mesure
de l’échange entre les participants. En dehors des tâches d’extraction d’informations telles que la
classiﬁcation de conversations (Koço et al., 2012) ou le résumé automatique (Trione et al., 2016),
```

#### Extractor Output:
- **fr_abstract:** corpus de conversation de type “chat” entre des clients et des opérateurs. Extraire des connaissances dans ce contexte demeure un déﬁpour les méthodes de traitement automatique des langues de par la dimension interactive et les propriétés de ce nouveau type de langage à l’intersection du langage écrit et parlé. Nous présentons une étude utilisant des réponses à des sondages utilisateurs comme supervision faible permettant de prédire la satisfaction des usagers d’un service en ligne d’assistance technique et commerciale.
- **en_abstract:** This paper presents methods to perform knowledge extraction from very large databases of WEB chat conversations between operators and clients in customer contact centers. Extracting knowledge from chat corpus is a challenging research issue. Simply applying traditional text mining tools is clearly sub-optimal as it takes into account neither the interaction dimension nor the particular nature of this language which shares properties of both spoken and written language. We present a method predicting users satisfaction in a chat-based service trained on answers from users to satisfaction surveys.
- **body (first 300 chars):** Évaluation automatique de la satisfaction client à partir de conversations de type "chat" par réseaux de neurones récurrents avec mécanisme d’attention Jeremy Auguste1 Delphine Charlet2 Géraldine Damnati2 Benoit Favre1 Frédéric Béchet1 (1) Aix-Marseille Univ, Université de Toulon, CNRS, LIS, Marseil

---

### Paper: `2015.jeptalnrecital-court.13`
- **PDF Path:** `data\raw\extended\2015.jeptalnrecital-court.13.pdf`
#### Where did the first words of the French abstract go?
**Preserved (NOT dropped).**
- Raw page 1 text has `Résumé.\nDans cet article, nous proposons une méthode de résumé automatique...`.
- Because of the trailing period `.` in `Résumé.`, the regex `(?im)^\s*(?:résumé|resume)\s*:?.*$` matched the period with `.*$` and did *not* match `\n` across lines.
- The first line was preserved completely: `Dans cet article, nous proposons une méthode...`.
- However, the English abstract contains the English title at its beginning (`Automatic Summarization Using a Genetic Algorithm`), and the body begins with the unstripped conference header line: `22ème Traitement Automatique des Langues Naturelles, Caen, 2015`.

```text
--- REGEX MATCH DETAILS ---
FR Header Match (368:375): 'Résumé.'
EN Header Match (996:1005): 'Abstract.'
```

#### Raw Page 1 Text (first 2500 characters):
```text
22ème Traitement Automatique des Langues Naturelles, Caen, 2015
Une Approche évolutionnaire pour le résumé automatique
Aurélien Bossard1
Christophe Rodrigues2
(1) Université Paris 8, Laboratoire d’Informatique Avancée de Saint-Denis
(2) CNRS, UMR 7030, Laboratoire d’Informatique de Paris Nord
(1) aurelien.bossard@iut.univ-paris8.fr, (2) christophe.rodrigues@lipn.fr
Résumé.
Dans cet article, nous proposons une méthode de résumé automatique fondée sur l’utilisation d’un al-
gorithme génétique pour parcourir l’espace des résumés candidats couplé à un calcul de divergence de distribution de
probabilités de n-grammes entre résumés candidats et documents source. Cette méthode permet de considérer un résumé
non plus comme une accumulation de phrases indépendantes les unes des autres, mais comme un texte vu dans sa globa-
lité. Nous la comparons à une des meilleures méthodes existantes fondée sur la programmation linéaire en nombre entier,
et montrons son efﬁcacité sur le corpus TAC 2009.
Abstract.
Automatic Summarization Using a Genetic Algorithm
This paper proposes a novel method for automatic summarization based on a genetic algorithm that explores candidate
summaries space following an objective function computed over ngrams probability distributions of the candidate sum-
mary and the source documents. This method does not consider a summary as a stack of independant sentences but as a
whole text. We compare this method to one of the best existing methods which is based on integer linear programming,
and show its efﬁciency on TAC 2009 corpus.
Mots-clés :
Résumé automatique, algorithme génétique, modèles probabilistes.
Keywords:
automatic summarization, genetic algorithm, probabilistic models.
1
Introduction
Les systèmes de résumé automatique (RA) sont des constituants essentiels des systèmes d’information. En effet, La
multiplication des sources d’information numérique rend parfois difﬁcile la lecture et l’assimilation d’un contenu en
ligne, même dans le cas où celui-ci est issu d’une recherche précise. Résumer automatiquement ces contenus peut alors
proposer une nouvelle approche d’un contenu informationnel. Le RA est donc naturellement devenu une des premières
thématiques de recherche en traitement automatique du langage (Luhn, 1958), et reste encore aujourd’hui un domaine
largement étudié.
Aﬁn de pouvoir valider les améliorations apportées par des changements de méthode ou de paramètres dans un système de
RA, il est nécessaire de disposer de méthodes d’éval
```

#### Extractor Output:
- **fr_abstract:** Dans cet article, nous proposons une méthode de résumé automatique fondée sur l’utilisation d’un al- gorithme génétique pour parcourir l’espace des résumés candidats couplé à un calcul de divergence de distribution de probabilités de n-grammes entre résumés candidats et documents source. Cette méthode permet de considérer un résumé non plus comme une accumulation de phrases indépendantes les unes des autres, mais comme un texte vu dans sa globa- lité. Nous la comparons à une des meilleures méthodes existantes fondée sur la programmation linéaire en nombre entier, et montrons son efﬁcacité sur le corpus TAC 2009.
- **en_abstract:** Automatic Summarization Using a Genetic Algorithm This paper proposes a novel method for automatic summarization based on a genetic algorithm that explores candidate summaries space following an objective function computed over ngrams probability distributions of the candidate sum- mary and the source documents. This method does not consider a summary as a stack of independant sentences but as a whole text. We compare this method to one of the best existing methods which is based on integer linear programming, and show its efﬁciency on TAC 2009 corpus.
- **body (first 300 chars):** 22ème Traitement Automatique des Langues Naturelles, Caen, 2015 Aurélien Bossard1 Christophe Rodrigues2 (1) Université Paris 8, Laboratoire d’Informatique Avancée de Saint-Denis (2) CNRS, UMR 7030, Laboratoire d’Informatique de Paris Nord (1) aurelien.bossard@iut.univ-paris8.fr, (2) christophe.rodri

---

### Paper: `2021.jeptalnrecital-deft.7`
- **PDF Path:** `data\raw\extended\2021.jeptalnrecital-deft.7.pdf`
#### Where did the first words of the French abstract go?
**Dropped by the header regex rule + layout text block ordering.**
- Raw page 1 text has `RESUME \nCet article décrit la participation de l'équipe Nantalco à la tâche 2 du Défi Fouille de Textes 2021 \n(DEFT) : évaluation...`.
- The header regex matched `RESUME \n` and swallowed the entire first line `Cet article décrit la participation de l'équipe Nantalco à la tâche 2 du Défi Fouille de Textes 2021`.
- The extracted French abstract starts with `(DEFT) : évaluation automatique de copies...`.
- Furthermore, in the English abstract, the English title was broken across lines (`Textual \nsimilarity \n...`). The `Abstract` regex matched `ABSTRACT \nTextual `, dropping `Textual ` and leaving `similarity measurement for automatic evaluation...` (starting with lowercase).
- In addition, PyMuPDF extracted the authors first, then abstracts, and placed the French title at the very bottom of the page stream.

```text
--- REGEX MATCH DETAILS ---
FR Header Match (153:262): "RESUME \nCet article décrit la participation de l'équipe Nantalco à la tâche 2 du Défi Fouille de Textes 2021 "
EN Header Match (1180:1198): 'ABSTRACT \nTextual '
```

#### Raw Page 1 Text (first 2500 characters):
```text
Xiaoou Wang 1 Xingyu Liu 1 Yimei Yue 2 
(1) Université Paris Nanterre, France 
(2) Inalco, France 
xiaoouwangfrance@gmail.com 
RESUME 
Cet article décrit la participation de l'équipe Nantalco à la tâche 2 du Défi Fouille de Textes 2021 
(DEFT) : évaluation automatique de copies d'après une référence existante. Nous avons utilisé 
principalement des traits basés sur la similarité cosinus des deux vecteurs représentant la similarité 
textuelle entre des réponses d'étudiant et la référence. Plusieurs types de vecteurs ont été utilisés 
(vecteur d'occurrences de mots, vecteur tf-idf, embeddings non contextualisés de fastText, 
embeddings contextualisés de CamemBERT et enfin Sentence Embeddings Multilingues ajustés sur 
des corpus multilingues). La meilleure performance du concours sur cette tâche a été de 0.682 
(précision) et celle de notre équipe 0.639.  Cette performance a été obtenue avec les Sentence 
Embeddings Multilingues alors que celle des embeddings non ajustés ne s'est élevée qu'à 0.55, 
suggérant que de récents modèles de langues pré-entraînés doivent être fine-tunés afin d'avoir des 
embeddings adéquats au niveau phrastique. 
ABSTRACT 
Textual 
similarity 
measurement 
for 
automatic 
evaluation 
of 
students' 
answers 
This paper describes the Nantalco team's participation in task 2 of the DEFT contest in 2021: 
Automatic evaluation of students' answers based on an existing reference. We mainly used features 
based on the cosine similarity of the two vectors representing the textual similarity between student 
responses and the reference. Several types of vectors were used (count vector, tf-idf vector, non-
contextualized embeddings from fastText, contextualized embeddings from Camembert and fine-
tuned Multilingual Sentence Embeddings). The best performance of the contest on this task was 0.682 
(precision). The maximum performance of our team (0.639) was obtained with the Multilingual 
Sentence Embeddings, while the best performance of raw embeddings of CamemBERT was only 
0.55, suggesting that recent pre-trained language models need to be fine-tuned in order to have 
adequate embeddings at the sentence level. 
 
MOTS-CLES : évaluation automatique, similarité textuelle, CamemBERT 
KEYWORDS:   automatic evaluation, textual similarity, CamemBERT 
Mesure de similarité textuelle pour l'évaluation automatique de 
copies d'étudiants
Actes de la 28e Conférence sur le Traitement Automatique des Langues Naturelles
Lille, Franc
```

#### Extractor Output:
- **fr_abstract:** (DEFT) : évaluation automatique de copies d'après une référence existante. Nous avons utilisé principalement des traits basés sur la similarité cosinus des deux vecteurs représentant la similarité textuelle entre des réponses d'étudiant et la référence. Plusieurs types de vecteurs ont été utilisés (vecteur d'occurrences de mots, vecteur tf-idf, embeddings non contextualisés de fastText, embeddings contextualisés de CamemBERT et enfin Sentence Embeddings Multilingues ajustés sur des corpus multilingues). La meilleure performance du concours sur cette tâche a été de 0.682 (précision) et celle de notre équipe 0.639. Cette performance a été obtenue avec les Sentence Embeddings Multilingues alors que celle des embeddings non ajustés ne s'est élevée qu'à 0.55, suggérant que de récents modèles de langues pré-entraînés doivent être fine-tunés afin d'avoir des embeddings adéquats au niveau phrastique.
- **en_abstract:** similarity measurement for automatic evaluation of students' answers This paper describes the Nantalco team's participation in task 2 of the DEFT contest in 2021: Automatic evaluation of students' answers based on an existing reference. We mainly used features based on the cosine similarity of the two vectors representing the textual similarity between student responses and the reference. Several types of vectors were used (count vector, tf-idf vector, non- contextualized embeddings from fastText, contextualized embeddings from Camembert and fine- tuned Multilingual Sentence Embeddings). The best performance of the contest on this task was 0.682 (precision). The maximum performance of our team (0.639) was obtained with the Multilingual Sentence Embeddings, while the best performance of raw embeddings of CamemBERT was only 0.55, suggesting that recent pre-trained language models need to be fine-tuned in order to have adequate embeddings at the sentence level.
- **body (first 300 chars):** Xiaoou Wang 1 Xingyu Liu 1 Yimei Yue 2 (1) Université Paris Nanterre, France (2) Inalco, France xiaoouwangfrance@gmail.com MOTS-CLES : évaluation automatique, similarité textuelle, CamemBERT KEYWORDS: automatic evaluation, textual similarity, CamemBERT Mesure de similarité textuelle pour l'évaluatio

---

### Paper: `2016.jeptalnrecital-jep.65`
- **PDF Path:** `data\raw\extended\2016.jeptalnrecital-jep.65.pdf`
#### Where did the first words of the French abstract go?
**Dropped by the header regex rule.**
- Raw page 1 text has `RESUME \nLa performance imitative des locuteurs varie de celle du professionnel, expert, à celle du naïf, plus \nou moins talentueux...`.
- The regex `(?im)^\s*(?:résumé|resume)\s*:?.*$` consumed `RESUME \n` plus the full line `La performance imitative des locuteurs varie de celle du professionnel, expert, à celle du naïf, plus `.
- The extracted French abstract starts mid-sentence with: `ou moins talentueux. L’étude de l’imitation souligne...`.

```text
--- REGEX MATCH DETAILS ---
FR Header Match (239:349): 'RESUME \nLa performance imitative des locuteurs varie de celle du professionnel, expert, à celle du naïf, plus '
EN Header Match (1224:1305): 'ABSTRACT \nWhich measure(s) of prosodic similarity as an evaluation of imitation? '
```

#### Raw Page 1 Text (first 2500 characters):
```text
Quelle(s) mesure(s) de similarité prosodique comme évaluation de 
l’imitation ? 
Olivier Nocaudie Corine Astésano 
 
U.R.I. Octogone-Lordat (E.A. 4156), Université de Toulouse, UTM 
 
nocaudie@univ-tlse2.fr, corine.astesano@univ-tlse2.fr 
RESUME 
La performance imitative des locuteurs varie de celle du professionnel, expert, à celle du naïf, plus 
ou moins talentueux. L’étude de l’imitation souligne la difficulté pour trouver des indices 
mesurables de la réussite d’une imitation. Dans cette étude exploratoire, des contours de f0 recueillis 
au fil de tâches d’imitation sont testés au moyen d’une double approche : mesure objective par le 
biais de deux mesures de la similarité prosodique reportées dans la littérature et évaluation 
perceptive par un panel de 15 auditeurs naïfs. Nos premiers résultats indiquent une bonne 
corrélation entre les deux approches et soulèvent la question du choix de l’indice mesurable qui 
rendrait le mieux compte d’une imitation au niveau tonal. Ils soulignent également la variabilité 
interindividuelle des comportements imitatifs en parole tout en ouvrant des perspectives 
intéressantes dans le domaine de la formation à la phonétique corrective par la Méthode Verbo-
tonale. 
ABSTRACT 
Which measure(s) of prosodic similarity as an evaluation of imitation? 
Imitative proficiency across speakers is highly variable. Studies on imitation underline how difficult 
it is to find measurable cues to assess a successful imitation. In this exploratory study, f0 contours 
collected from imitations tasks are tested in a double approach: objective measurements of prosodic 
similarity using two measures reported in the literature and perceptive evaluation by a panel of 15 
naïve listeners. Our first results indicate a good correlation between the two approaches and they 
raise the question concerning the selection of the measurable factor assessing a successful imitation 
at a tonal level. Furthermore, these results underline an imitative proficiency’s variability across 
speakers while opening perspectives in the domain of phonetic correction using the Verbo Tonal 
Method. 
 
MOTS-CLES :  parole, prosodie, imitation, mesures objectives, évaluations perceptives. 
KEYWORDS:   speech, prosody, imitation, objective measurements, perceptive evaluation. 
 
1 Introduction 
Les études sur l’imitation de la parole rapportent différents types de comportements imitatifs, 
comme la convergence (une adaptation  mutuelle des interlocuteurs au fil de la co
```

#### Extractor Output:
- **fr_abstract:** ou moins talentueux. L’étude de l’imitation souligne la difficulté pour trouver des indices mesurables de la réussite d’une imitation. Dans cette étude exploratoire, des contours de f0 recueillis au fil de tâches d’imitation sont testés au moyen d’une double approche : mesure objective par le biais de deux mesures de la similarité prosodique reportées dans la littérature et évaluation perceptive par un panel de 15 auditeurs naïfs. Nos premiers résultats indiquent une bonne corrélation entre les deux approches et soulèvent la question du choix de l’indice mesurable qui rendrait le mieux compte d’une imitation au niveau tonal. Ils soulignent également la variabilité interindividuelle des comportements imitatifs en parole tout en ouvrant des perspectives intéressantes dans le domaine de la formation à la phonétique corrective par la Méthode Verbo- tonale.
- **en_abstract:** Imitative proficiency across speakers is highly variable. Studies on imitation underline how difficult it is to find measurable cues to assess a successful imitation. In this exploratory study, f0 contours collected from imitations tasks are tested in a double approach: objective measurements of prosodic similarity using two measures reported in the literature and perceptive evaluation by a panel of 15 naïve listeners. Our first results indicate a good correlation between the two approaches and they raise the question concerning the selection of the measurable factor assessing a successful imitation at a tonal level. Furthermore, these results underline an imitative proficiency’s variability across speakers while opening perspectives in the domain of phonetic correction using the Verbo Tonal Method.
- **body (first 300 chars):** Quelle(s) mesure(s) de similarité prosodique comme évaluation de l’imitation ? Olivier Nocaudie Corine Astésano U.R.I. Octogone-Lordat (E.A. 4156), Université de Toulouse, UTM nocaudie@univ-tlse2.fr, corine.astesano@univ-tlse2.fr MOTS-CLES : parole, prosodie, imitation, mesures objectives, évaluatio

---

## 2. Quantitative Audit Across 130 Processed Papers

The 130 processed papers consist of:
- **20 Sources** (from `results/real/papers.jsonl` matching `data/processed/sources_final.json`)
- **10 Dev Papers** (from `data/processed/selected_metadata_new.jsonl` extracted from PDFs in `data/raw/`)
- **100 Eligible French Distractors** (from `data/processed/french_distractors_native_v2.jsonl` with `eligible: true`)

| Corpus Subset | N | (a) fr_abstract lacking initial cap | (a) fr_abstract starts lowercase | (b) en_abstract starts title/challenge | (c) body contains abstract[:80] | (d) body starts with French title |
|---|---|---|---|---|---|---|
| **Sources** | 20 | 18 (90.0%) | 17 (85.0%) | 4 (20.0%) | 1 (5.0%) | 16 (80.0%) |
| **Dev** | 10 | 8 (80.0%) | 8 (80.0%) | 2 (20.0%) | 0 (0.0%) | 9 (90.0%) |
| **Distractors** | 100 | 70 (70.0%) | 68 (68.0%) | 5 (5.0%) | 0 (0.0%) | 58 (58.0%) |
| **TOTAL** | **130** | **96 (73.8%)** | **93 (71.5%)** | **11 (8.5%)** | **1 (0.8%)** | **83 (63.8%)** |

### Detailed Observations on Question 2 Criteria:

1. **(a) French Abstract Truncation:** Across the entire set of 130 papers, **96/130 (73.8%)** lack an initial capital letter (and **93/130** explicitly start with lowercase). This proves that the `(?:résumé|resume)\s*:?.*$` newline consumption bug is ubiquitous across all years whenever 'Résumé' appears without a period on its own line.
2. **(b) English Abstract Title/Challenge Leaks:** **11/130 (8.5%)** English abstracts have the paper's English title or challenge label prepended because 'Abstract' was followed immediately by the title before the abstract body.
3. **(c) Abstract Text Inside Body:** Only **1/130** paper (`2017.jeptalnrecital-court.2`) contains the first 80 characters of the abstract inside the body. This occurred not because of an extraction failure to remove the abstract on page 1, but because the authors verbatim repeated the abstract sentence in their discussion section (`c.-à-d., une mesure d’association attribue-t-elle des scores élevés à des n-grammes...`).
4. **(d) Body Starting With Title:** **83/130 (63.8%)** of all bodies start directly with the French title. The remaining papers fail to start with the title because the extractor captured running conference header text (`22ème Traitement Automatique...`, `TALN 2009...`), or author names were extracted before the title due to multi-column text-flow ordering in the PDF.

---

## 3. Audit of Blind Human Task Inputs

We evaluated `data/human/t3_input.jsonl`, `data/human/t4_passA_input.jsonl`, and `data/human/t4_passB_input.jsonl`.

### Summary of Leaks by File:

#### File 1: `data/human/t3_input.jsonl`
- **Purpose:** Input for Task 3 (human query generation from abstracts).
- **Fields:** `['paper_id', 'abstract_en', 'abstract_fr']`
- **Leaks Detected:**
  1. **English Title Leak:** For papers where `abstract_en` starts with the title, the English title is directly exposed:
     - `2015.jeptalnrecital-court.13`: leaks `Automatic Summarization Using a Genetic Algorithm` (first 6 words).
     - `2015.jeptalnrecital-recital.3`: leaks `Toward a detection model of affects, appreciations, jugements within human-agent interactions` (first 10 words).
     - `2021.jeptalnrecital-deft.1`: leaks `DEFT 2021 Challenge` (challenge label).
  2. **French Abstract Truncation:** 8 of the 10 papers have French abstracts starting mid-sentence, providing corrupted semantic context to human writers.

#### File 2: `data/human/t4_passA_input.jsonl`
- **Purpose:** Pass A of the Facet Coverage Audit (supposed to judge facets from English abstract).
- **Fields:** `['paper_id', 'abstract_en']`
- **Leaks Detected:**
  - Contains the identical English title leaks as `t3_input.jsonl` in `2015.jeptalnrecital-court.13`, `2015.jeptalnrecital-recital.3`, and `2021.jeptalnrecital-deft.1`.
  - Evaluators reading `abstract_en` are biased by having the exact English title placed at the start of the abstract text.

#### File 3: `data/human/t4_passB_input.jsonl`
- **Purpose:** Pass B of the Facet Coverage Audit (supposed to judge facets blindly from French body text).
- **Fields:** `['paper_id', 'body_fr']`
- **Leaks Detected:**
  1. **Full English Abstract in `t4_passB`?** -> **No full English abstract leak.** The extractor successfully removed the `abstract_en` section.
  2. **English Keywords Leak:** **10/10 (100%)** of records in `t4_passB_input.jsonl` contain the `KEYWORDS:` block with full English keyword phrases!
     - E.g. `KEYWORDS: neural model, transfer learning, transcription, Italian Comedy.`
     - E.g. `KEYWORDS: lexical semantic network, inference, multilinguality.`
     - E.g. `Keywords: automatic summarization, genetic algorithm, probabilistic models.`
  3. **Metadata & Author Leak:** The extractor left the entire page-1 front matter at the start of `body_fr`:
     - French title, author names, affiliations, email addresses, and `MOTS-CLÉS` are all present before the Introduction.
     - This destroys blinding for human evaluators in Pass B, allowing them to see author identities and English topic keywords.

### Itemized Leak Table for `data/human/` Files:

| Paper ID | `t3_input.jsonl` Title Leak | `t4_passA_input.jsonl` Title Leak | `t4_passB_input.jsonl` Keywords / Front Matter Leak |
|---|---|---|---|
| `2018.jeptalnrecital-long.14` | None | None | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2019.jeptalnrecital-court.18` | None | None | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2015.jeptalnrecital-court.13` | `Automatic Summarization Using a Genetic Algorithm` | `Automatic Summarization Using a Genetic Algorithm` | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2017.jeptalnrecital-court.5` | None | None | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2023.jeptalnrecital-long.11` | None | None | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2015.jeptalnrecital-recital.3` | `Toward a detection model of affects, appreciations, jugements...` | `Toward a detection model of affects, appreciations, jugements...` | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2021.jeptalnrecital-taln.4` | None | None | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2020.jeptalnrecital-taln.1` | None | None | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2021.jeptalnrecital-deft.1` | `DEFT 2021 Challenge` | `DEFT 2021 Challenge` | Leaks English `KEYWORDS:`, French title, authors, affiliations |
| `2019.jeptalnrecital-court.7` | None | None | Leaks English `KEYWORDS:`, French title, authors, affiliations |