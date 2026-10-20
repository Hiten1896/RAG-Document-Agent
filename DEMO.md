# Demo script

1. Start the backend and frontend, then upload `eval/grounding_test_doc.txt`.
2. Ask “Who founded the Zorvania Institute and when?” and open **Show evidence**.
3. Ask the water-boiling contradiction question and show that the answer follows
   the document's stated 60°C value.
4. Ask “What is the capital of France?” and show the exact not-found response.
5. Point to the **Answered by provider / model** line and the retrieved chunks,
   sources, and scores in the evidence panel.
6. Open `eval/results.md` and show the legacy, strict, automatic-routing, and
   no-context comparison results.

Uploads are held in an in-memory ChromaDB collection and are lost when the
backend restarts.
