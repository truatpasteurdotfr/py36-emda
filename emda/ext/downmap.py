import os, gzip, shutil
import urllib.request
from contextlib import closing
import xml.etree.ElementTree as ET


def fetch_data(emdbid):
    headerxml = (
        "ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/header/emd-%s.xml"
        % (emdbid, emdbid)
    )
    header30xml = (
        "ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/header/emd%s-v30.xml"
        % (emdbid, emdbid)
    )

    outmap = None; outcif = None; claimed_resol = None

    with closing(urllib.request.urlopen(headerxml)) as r:
        with open("file.xml", "wb") as f:
            shutil.copyfileobj(r, f)

    tree = ET.parse("file.xml")
    root = tree.getroot()

    model = None
    for deposition in root.findall("deposition"):
        for fittedPDBEntryIdList in deposition.findall("fittedPDBEntryIdList"):
            for fittedPDBEntryId in fittedPDBEntryIdList.findall("fittedPDBEntryId"):
                model = fittedPDBEntryId.text

    for processing in root.findall("processing"):
        for reconstruction in processing.findall("reconstruction"):
            for resolutionByAuthor in reconstruction.findall("resolutionByAuthor"):
                claimed_resol = resolutionByAuthor.text

    # create a directory with EMDBID
    #path = "/Users/sandy/MRC/REFMAC/COVID19/EMD-%s/" % (emdbid)
    path = os.getcwd() + "/EMD-%s/" % (emdbid)
    try:
        os.mkdir(path)
    except OSError:
        if os.path.isdir(path):
            print("Directory exists!")
        else:
            print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s " % path)

    if model is not None:
        try:
            cifftplink = "ftp://ftp.ebi.ac.uk/pub/databases/pdb/data/structures/all/mmCIF/{}.cif.gz".format(
                model
            )
            outcif = "%s.cif" % (model)
        except Exception as e:
            print(e)
        try:
            with urllib.request.urlopen(cifftplink) as response:
                with gzip.GzipFile(fileobj=response) as uncompressed:
                    file_content = uncompressed.read()
            with open(path + outcif, "wb") as f:
                f.write(file_content)
        except Exception as e:
            print(e)
    mapftplink = (
        "ftp://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-%s/map/emd_%s.map.gz"
        % (emdbid, emdbid)
    )

    outmap = "emd_%s.map" % (emdbid)
    try:
        with urllib.request.urlopen(mapftplink) as response:
            with gzip.GzipFile(fileobj=response) as uncompressed:
                file_content = uncompressed.read()
        with open(path + outmap, "wb") as f:
            f.write(file_content)
    except Exception as e:
        print(e)

def main(emdbidList):
    for emdbid in emdbidList:
        fetch_data(emdbid)
        print("Fetched ", emdbid)

""" if __name__ == "__main__":
    emdbidList = ["11310"]#, "11276", "11288", "11321", "11325"]
    for emdbid in emdbidList:
        fetch_data(emdbid)
    print("Fetched data") """