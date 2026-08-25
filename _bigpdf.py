import fitz, os
doc = fitz.open()
para = ("Computer organization and architecture concerns the operational units and "
        "their interconnections that realize the architectural specifications. "
        "Cache memory, pipelining, and instruction-level parallelism are core topics. ") * 6
for i in range(400):
    p = doc.new_page()
    p.insert_textbox(fitz.Rect(40, 40, 560, 780), f"Chapter page {i+1}\n\n" + para, fontsize=9)
doc.save("_big.pdf")
print("pages:", doc.page_count, "bytes:", os.path.getsize("_big.pdf"), "MB: %.2f" % (os.path.getsize("_big.pdf")/1048576))
