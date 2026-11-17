# Sample grounded answers (Part 4)

k = 5, chunk size = 500, overlap = 100

## Q: What is overfitting and how can it be detected?

**Model:** gemini-3.5-flash-lite

Overfitting occurs when a model is fit too closely to the particularities of the training set, allowing it to perform well on the training data but failing to generalize to new data [2]. 

It can be detected by comparing the model's performance on the training set versus the test set, specifically when there is a large discrepancy where the training set score is much higher than the test set score [3].

**Sources:** [1] troubleshooting_runbook.pdf p6 (score 0.0405); [2] intro-to-ml-book.pdf p42 (score 0.036); [3] intro-to-ml-book.pdf p62 (score 0.0348); [4] intro-to-ml-book.pdf p97 (score 0.025); [5] intro-to-ml-book.pdf p63 (score 0.0232)
**Grounding:** `{"lexical_support": 0.9, "citations": [2, 3], "invalid_citations": [], "refused": false}`

## Q: What is regularization used for?

**Model:** gemini-3.5-flash-lite

Regularization is used to explicitly restrict a model to avoid overfitting [3].

**Sources:** [1] intro-to-ml-book.pdf p66 (score 0.0799); [2] intro-to-ml-book.pdf p68 (score 0.067); [3] intro-to-ml-book.pdf p63 (score 0.066); [4] intro-to-ml-book.pdf p74 (score 0.0653); [5] intro-to-ml-book.pdf p69 (score 0.0646)
**Grounding:** `{"lexical_support": 0.714, "citations": [3], "invalid_citations": [], "refused": false}`

## Q: How do I fix an ERR-401 expired token error?

**Model:** gemini-3.5-flash-lite

To fix an ERR-401 expired token error:
1. Restore the previous TTL (default 900 seconds) [2].
2. Force a JWKS refresh by restarting the consumer pods one at a time [2].
3. Fix NTP drift if present [2].
4. Ask client teams to enable proactive refresh 60 seconds before expiry [2].

**Sources:** [1] troubleshooting_runbook.pdf p6 (score 0.152); [2] troubleshooting_runbook.pdf p2 (score 0.1481); [3] troubleshooting_runbook.pdf p5 (score 0.1271); [4] troubleshooting_runbook.pdf p3 (score 0.0605); [5] troubleshooting_runbook.pdf p1 (score 0.0323)
**Grounding:** `{"lexical_support": 1.0, "citations": [2], "invalid_citations": [], "refused": false}`

## Q: How do I renew an expired TLS certificate?

**Model:** gemini-3.5-flash-lite

To renew an expired TLS certificate, follow these steps:

* Renew the certificate through cert-manager by running: `kubectl cert-manager renew <certificate-name>` [1][2].
* Reload the ingress controller [1][2].
* Verify the full chain [1][2].
* If the ACME challenge fails, check your DNS-01 credentials [1][2].

**Sources:** [1] troubleshooting_runbook.pdf p5 (score 0.1393); [2] troubleshooting_runbook.pdf p6 (score 0.0992); [3] troubleshooting_runbook.pdf p2 (score 0.0155); [4] intro-to-ml-book.pdf p10 (score 0.0); [5] intro-to-ml-book.pdf p9 (score 0.0)
**Grounding:** `{"lexical_support": 0.897, "citations": [1, 2], "invalid_citations": [], "refused": false}`

## Q: What is the best recipe for sourdough bread?

**Model:** gemini-3.5-flash-lite

I don't have enough information in the provided documents to answer that.

**Sources:** [1] intro-to-ml-book.pdf p314 (score 0.0593); [2] intro-to-ml-book.pdf p276 (score 0.0198); [3] intro-to-ml-book.pdf p275 (score 0.0195); [4] intro-to-ml-book.pdf p279 (score 0.019); [5] intro-to-ml-book.pdf p280 (score 0.0181)
**Grounding:** `{"lexical_support": 0.2, "citations": [], "invalid_citations": [], "refused": true}`
