"""
This file contains the code that loads all of our information in as JSONs
"""
import json

def convert_pogp():
    pogp = {}
    real_pogp = []
    with open("datasets/pogp.json",'r') as f:
        pogp = json.loads(f.read())
    books = pogp['books']
    for book in books:
        chapters = book['chapters']
        for chapter in chapters:
            verses = chapter["verses"]
            for verse in verses:
                real_pogp.append({"reference":verse["reference"],
                                "text" : verse["text"],
                                "chapter" : chapter["reference"],
                                "book" : book["book"]
                })
    with open("datasets/pogp1.json","w") as f:
        json.dump(real_pogp,f,indent=4)
def convert_dc():
    d_c = {}
    real_d_c = []
    with open("datasets/d_c.json",'r') as f:
        d_c = json.loads(f.read())
    sections = d_c["sections"]
    for section in sections:
        verses = section['verses']
        for verse in verses:
            real_d_c.append({
                "reference" : verse["reference"],
                "text" : verse["text"],
                "chapter" : section["reference"],
                "book" : "D&C"
            })
    with open("datasets/dc.json","w") as f:
        json.dump(real_d_c,f,indent = 4)
def convert_bom_bible():
    bom = {}
    real_bom = []
    with open("datasets/bookOfMormon.json",'r') as BoM_file:
        bom = json.loads(BoM_file.read())
    books = bom['books']
    for book in books:
        chapters = book['chapters']
        for chapter in chapters:
            verses = chapter["verses"]
            for verse in verses:
                real_bom.append({"reference":verse["reference"],
                                "text" : verse["text"],
                                "chapter" : chapter["reference"],
                                "book" : book["book"]
                })
    bible = {}
    real_bible = []
    with open("datasets/bible.json",'r') as bible_file:
        bible = json.loads(bible_file.read())
    se = set()
    for verse in bible:
        ref = verse["name"]
        index = ref.rfind(':')
        replace = True
        book_name = ref[:index]
        while replace:
            replace = False
            if book_name[-1].isnumeric():
                replace = True
                book_name = book_name[:-1]
        se.add(book_name[:-1])
        real_bible.append({
            "reference" : verse["name"],
            "text" : verse["verse"],
            "chapter" : ref[:index],
            "book" : book_name[:-1]
        })
    with open("datasets/bible1.json","w") as f:
        json.dump(real_bible,f,indent = 4)
    with open("datasets/bom.json","w") as f:
        json.dump(real_bom,f,indent = 4)
def all_scripture():
    json_files = ["bible.json","bom.json","dc.json","pogp.json"]

    all_scripture = {}
    for file_name in json_files:
        with open("datasets/"+file_name,'r') as f:
            file = json.loads(f.read())
        for p in file:
            all_scripture[p["reference"]] = p["text"]
    with open("datasets/all_references.json","w") as file:
        json.dump(all_scripture, file, indent=4)