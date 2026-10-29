from __future__ import annotations

from .corpus import Document


def documents(n: int = 20) -> list[Document]:
    result = []
    for number in range(n):
        topic = ["summarization", "translation", "classification", "speech"][number % 4]
        result.append(Document(
            doc_id=f"fr-{number:03d}", kind="source", lang="fr",
            title=f"Etude {topic} multilingue {number}",
            abstract_fr=f"Cette étude propose une méthode pour la {topic} et mesure une amélioration.",
            abstract_en=f"This study proposes a method for multilingual {topic} and measures an improvement.",
            body_fr=f"Nous analysons le problème de {topic}. La méthode computationnelle utilise des données annotées. "
                    f"Les expériences évaluent la qualité et montrent une amélioration mesurable. " * 100,
            year=2015 + number % 10,
        ))
    result.extend(Document(
        doc_id=f"en-{number:03d}", kind="en_distractor", lang="en",
        title=f"English unrelated study {number}", abstract_en="This paper studies a different topic.",
    ) for number in range(max(4, n // 4)))
    return result
