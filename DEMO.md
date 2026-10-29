# Demo script

1. Start the backend and frontend, then upload `eval/grounding_test_doc.txt`.
2. Click **Mind map** beside **Executive summary**. Explore the map by zooming,
   panning, and collapsing branches; click a node to show its source location.
   Try **Fit to screen**, **Download SVG**, **Download PNG**, and **Regenerate**.
3. Ask “Who founded the Zorvania Institute and when?” and open **Show evidence**.
   Use the mind-map button under the answer to create a topic-focused map.
4. Click the answer's source card or inline filename/page citation. Confirm that
   the PDF opens at its cited page. For a text-based upload, confirm the viewer
   shows the extracted passage and location; PDF text highlighting is not
   included.
5. Ask the water-boiling contradiction question and show that the answer follows
   the document's stated 60°C value.
6. Ask “What is the capital of France?” and show the exact not-found response.
7. Point to the **Answered by provider / model** line and the retrieved chunks,
   sources, and scores in the evidence panel.
8. Click **Suggest questions** below an answer. Confirm three follow-up chips
   appear, then click one to ask it as the next question. The exact not-found
   response does not offer suggestions.
9. Open `eval/results.md` and show the legacy, strict, automatic-routing, and
   no-context comparison results.

Uploads are held in an in-memory ChromaDB collection and are lost when the
backend restarts.
