import { PaperDocument } from "./types";

export const MOCK_PAPER: PaperDocument = {
  id: "paper-001",
  title:
    "Improving Retrieval-Augmented Generation with Evidence-Linked Reranking",
  titleKo: "증거 연결 재순위화를 통한 검색 증강 생성 개선",
  authors: ["J. Kim", "S. Park", "M. Lee"],
  source: "arXiv:2505.09388",
  sections: [
    {
      id: "sec-abstract",
      title: "Abstract",
      titleKo: "초록",
      paragraphs: [
        {
          id: "P0",
          spans: [
            {
              id: "P0.S1",
              original:
                "Retrieval-Augmented Generation (RAG) systems often struggle with context precision when the retrieved passages contain distracting or irrelevant information.",
              translated:
                "검색 증강 생성(RAG) 시스템은 검색된 passage가 산만하거나 관련 없는 정보를 포함할 때 컨텍스트 정밀도에 어려움을 겪는 경우가 많다.",
            },
            {
              id: "P0.S2",
              original:
                "We propose an evidence-linked reranking method that improves top-k context precision by explicitly scoring passages based on their evidentiary connection to the query.",
              translated:
                "우리는 query와의 증거적 연결성을 기반으로 passage를 명시적으로 점수화하여 top-k 컨텍스트 정밀도를 향상시키는 증거 연결 재순위화 방법을 제안한다.",
            },
            {
              id: "P0.S3",
              original:
                "Our approach reduces irrelevant context injection by 34% while maintaining recall on standard benchmarks.",
              translated:
                "우리의 접근 방식은 표준 벤치마크에서 재현율을 유지하면서 관련 없는 컨텍스트 주입을 34% 줄인다.",
            },
          ],
        },
      ],
    },
    {
      id: "sec-intro",
      title: "1. Introduction",
      titleKo: "1. 서론",
      paragraphs: [
        {
          id: "P1",
          spans: [
            {
              id: "P1.S1",
              original:
                "The fundamental challenge in RAG systems is balancing recall and precision in the retrieved context.",
              translated:
                "RAG 시스템의 근본적인 과제는 검색된 컨텍스트에서 재현율과 정밀도의 균형을 맞추는 것이다.",
            },
            {
              id: "P1.S2",
              original:
                "Retrieving more passages improves recall but introduces noise that can mislead the generator, a phenomenon we call context dilution.",
              translated:
                "더 많은 passage를 검색하면 재현율은 향상되지만 생성기를 오도할 수 있는 노이즈가 도입되며, 우리는 이를 컨텍스트 희석(context dilution)이라 부른다.",
            },
            {
              id: "P1.S3",
              original:
                "Previous work on passage reranking focused primarily on relevance scoring without considering whether the passage provides direct evidence for answering the query.",
              translated:
                "passage 재순위화에 대한 이전 연구는 passage가 query에 대한 직접적인 증거를 제공하는지 여부를 고려하지 않고 주로 관련성 점수화에 초점을 맞추었다.",
            },
          ],
        },
        {
          id: "P2",
          spans: [
            {
              id: "P2.S1",
              original:
                "We introduce Evidence-Linked Reranking (ELR), a method that assigns each candidate passage an evidence score by measuring the semantic overlap between the passage's factual claims and the information need expressed in the query.",
              translated:
                "우리는 각 후보 passage에 passage의 사실적 주장과 query에 표현된 정보 요구 간의 의미적 중첩을 측정하여 증거 점수를 부여하는 방법인 증거 연결 재순위화(ELR)를 소개한다.",
            },
            {
              id: "P2.S2",
              original:
                "Unlike traditional rerankers that produce a single relevance score, ELR decomposes the scoring into three components: factual alignment, evidential support, and contextual coherence.",
              translated:
                "단일 관련성 점수를 생성하는 기존 재순위기와 달리 ELR은 점수화를 세 가지 구성 요소로 분해한다: 사실 정렬(factual alignment), 증거 지원(evidential support), 문맥적 일관성(contextual coherence).",
            },
          ],
        },
      ],
    },
    {
      id: "sec-method",
      title: "2. Method",
      titleKo: "2. 방법론",
      paragraphs: [
        {
          id: "P3",
          spans: [
            {
              id: "P3.S1",
              original:
                "Given a query q and a set of candidate passages C = {c₁, c₂, ..., cₙ} retrieved by the first-stage retriever, ELR computes a composite evidence score for each passage.",
              translated:
                "query q와 1단계 검색기에 의해 검색된 후보 passage 집합 C = {c₁, c₂, ..., cₙ}이 주어지면, ELR은 각 passage에 대해 복합 증거 점수를 계산한다.",
            },
            {
              id: "P3.S2",
              original:
                "The factual alignment score measures whether the passage contains claims that directly address the query's information need, computed as the maximum cosine similarity between claim embeddings and the query embedding.",
              translated:
                "사실 정렬 점수는 passage가 query의 정보 요구를 직접적으로 다루는 주장을 포함하는지 측정하며, claim 임베딩과 query 임베딩 간의 최대 코사인 유사도로 계산된다.",
            },
            {
              id: "P3.S3",
              original:
                "The evidential support score evaluates whether the passage provides verifiable evidence rather than mere assertions, using a lightweight NLI classifier trained to distinguish supported claims from unsupported ones.",
              translated:
                "증거 지원 점수는 passage가 단순한 주장이 아닌 검증 가능한 증거를 제공하는지 평가하며, 지지된 주장과 지지되지 않은 주장을 구별하도록 훈련된 경량 NLI 분류기를 사용한다.",
            },
          ],
        },
        {
          id: "P4",
          spans: [
            {
              id: "P4.S1",
              original:
                "The final evidence score is a weighted combination: ELR(q, c) = α · FA(q, c) + β · ES(q, c) + γ · CC(q, c), where α, β, γ are learned weights.",
              translated:
                "최종 증거 점수는 가중 조합이다: ELR(q, c) = α · FA(q, c) + β · ES(q, c) + γ · CC(q, c), 여기서 α, β, γ는 학습된 가중치이다.",
            },
            {
              id: "P4.S2",
              original:
                "We optimize these weights on a held-out validation set using a pairwise ranking loss that encourages passages with stronger evidence to be ranked higher.",
              translated:
                "우리는 더 강한 증거를 가진 passage가 더 높은 순위에 오도록 유도하는 pairwise ranking loss를 사용하여 hold-out 검증 세트에서 이 가중치를 최적화한다.",
            },
          ],
        },
      ],
    },
    {
      id: "sec-experiments",
      title: "3. Experiments",
      titleKo: "3. 실험",
      paragraphs: [
        {
          id: "P5",
          spans: [
            {
              id: "P5.S1",
              original:
                "We evaluate ELR on three standard RAG benchmarks: Natural Questions (NQ), TriviaQA, and MS MARCO passage ranking.",
              translated:
                "우리는 세 가지 표준 RAG 벤치마크에서 ELR을 평가한다: Natural Questions(NQ), TriviaQA, MS MARCO passage ranking.",
            },
            {
              id: "P5.S2",
              original:
                "For each benchmark, we use BM25 and Contriever as first-stage retrievers and evaluate with top-k precision (k=5, 10, 20), recall@100, and downstream answer F1.",
              translated:
                "각 벤치마크에 대해 BM25와 Contriever를 1단계 검색기로 사용하고 top-k 정밀도(k=5, 10, 20), recall@100, downstream answer F1으로 평가한다.",
            },
          ],
        },
        {
          id: "P6",
          spans: [
            {
              id: "P6.S1",
              original:
                "ELR improves top-5 precision by 12.3% on NQ and 8.7% on TriviaQA over the Contriever baseline, while maintaining recall@100 within 1.2% of the unfiltered retrieval set.",
              translated:
                "ELR은 Contriever baseline 대비 NQ에서 top-5 정밀도를 12.3%, TriviaQA에서 8.7% 향상시키며, 필터링되지 않은 검색 집합 대비 recall@100을 1.2% 이내로 유지한다.",
            },
            {
              id: "P6.S2",
              original:
                "Notably, the evidential support component contributes the most to performance gains, suggesting that distinguishing evidence from assertion is more valuable than simple relevance matching.",
              translated:
                "특히 증거 지원 구성 요소가 성능 향상에 가장 크게 기여하며, 이는 증거와 주장을 구별하는 것이 단순한 관련성 매칭보다 더 가치 있음을 시사한다.",
            },
            {
              id: "P6.S3",
              original:
                "Ablation studies show that removing the evidential support component degrades top-5 precision by 7.1%, while removing factual alignment reduces it by 3.8%.",
              translated:
                "절삭 연구에 따르면 증거 지원 구성 요소를 제거하면 top-5 정밀도가 7.1% 저하되고, 사실 정렬을 제거하면 3.8% 감소한다.",
            },
          ],
        },
      ],
    },
    {
      id: "sec-discussion",
      title: "4. Discussion",
      titleKo: "4. 논의",
      paragraphs: [
        {
          id: "P7",
          spans: [
            {
              id: "P7.S1",
              original:
                "Our results suggest that current RAG systems over-rely on surface-level relevance signals and can benefit from explicit evidence reasoning.",
              translated:
                "우리의 결과는 현재 RAG 시스템이 표면 수준의 관련성 신호에 과도하게 의존하며 명시적 증거 추론으로부터 이점을 얻을 수 있음을 시사한다.",
            },
            {
              id: "P7.S2",
              original:
                "A key limitation of ELR is the added computational cost of the NLI classifier, which increases reranking latency by approximately 40ms per passage on a single GPU.",
              translated:
                "ELR의 핵심 한계는 NLI 분류기의 추가 계산 비용으로, 단일 GPU에서 passage당 재순위화 지연 시간이 약 40ms 증가한다.",
            },
            {
              id: "P7.S3",
              original:
                "Future work could explore distilling the evidence scoring into the first-stage retriever to eliminate the reranking overhead entirely.",
              translated:
                "향후 연구에서는 재순위화 오버헤드를 완전히 제거하기 위해 증거 점수화를 1단계 검색기로 증류하는 방법을 탐구할 수 있다.",
            },
          ],
        },
      ],
    },
    {
      id: "sec-conclusion",
      title: "5. Conclusion",
      titleKo: "5. 결론",
      paragraphs: [
        {
          id: "P8",
          spans: [
            {
              id: "P8.S1",
              original:
                "We presented Evidence-Linked Reranking, a method that improves RAG context precision by decomposing passage relevance into factual alignment, evidential support, and contextual coherence.",
              translated:
                "우리는 passage 관련성을 사실 정렬, 증거 지원, 문맥적 일관성으로 분해하여 RAG 컨텍스트 정밀도를 향상시키는 증거 연결 재순위화를 제시했다.",
            },
            {
              id: "P8.S2",
              original:
                "ELR reduces irrelevant context injection by 34% across three benchmarks while preserving recall, demonstrating that evidence-aware reranking is a practical and effective approach for improving RAG quality.",
              translated:
                "ELR은 재현율을 보존하면서 세 가지 벤치마크에서 관련 없는 컨텍스트 주입을 34% 줄여, 증거 인식 재순위화가 RAG 품질 향상을 위한 실용적이고 효과적인 접근 방식임을 보여준다.",
            },
          ],
        },
      ],
    },
  ],
};
