import sys
from stam import  AnnotationStore, Offset, Selector, TextSelectionOperator

def align(table: list[tuple[str,float,float]], reftext: str) -> AnnotationStore:
    text = " ".join((x[0] for x in table))
    store = AnnotationStore(id="tmp")
    transcription_resource = store.add_resource(text=text,id="transcription")
    
    #add word timing annotations on the transcribed text
    charbegin = 0
    annotationcount = 0
    for i, (word, starttime, endtime) in enumerate(table):
        annotationcount += 1
        if i > 0:
            charbegin += 1 #space
        charend = charbegin + len(word)
        store.annotate(target=Selector.textselector(transcription_resource, Offset.simple(charbegin, charend)), data=[{
            "set": "timeinfo",
            "key": "starttime", 
            "value": starttime,
        },{
            "set": "timeinfo",
            "key": "endtime", 
            "value": endtime,
        }
        ]) 
        charbegin = charend
    print(f"added {annotationcount} time annotations", file=sys.stderr)

    reference_resource = store.add_resource(text=reftext,id="reference")
    asr_textsel = transcription_resource.textselection(Offset.whole())
    ref_textsel = reference_resource.textselection(Offset.whole())
    transposed = 0
    for transposition in asr_textsel.align_texts(ref_textsel, case_sensitive=False, trim=True): #add grow=True for inprecise alignments
        #find all annotations covered by this alignment (a transposition)
        for left, right in transposition.alignments():
            for annotation in left.related_text(TextSelectionOperator.embeds()).annotations(set="timeinfo"):
                annotation.transpose(transposition)
                transposed += 1
            print(f"\"{left.text()}\"\t{left.offset()} -->  \"{right.text()}\"\t{right.offset()}")
    print(f"transposed {transposed} annotations", file=sys.stderr)
    return store








