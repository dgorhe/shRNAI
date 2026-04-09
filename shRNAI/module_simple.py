import numpy as np


def pair(seq1: str, seq2: str, index: int) -> int:
    # make pair information between upper and lower strand of pri-miRNA
    # includes U-G pair
    if index < 12:
        return 0
    if seq1 == "A" and seq2 == "T":
        return 1
    if seq1 == "T" and seq2 in ["A", "G"]:
        return 1
    if seq1 == "C" and seq2 == "G":
        return 1
    if seq1 == "G" and seq2 in ["C", "T"]:
        return 1
    return 0


def convert(
    seq: str,
) -> tuple[list[str], list[str], np.ndarray, np.ndarray]:
    # make inputs from sequence
    # both mature-shRNA and pri-shRNA
    trans = str.maketrans("ACGT", "0123")
    trans2 = str.maketrans("ACGT", "TGCA")

    seqK: list[str] = []
    priK: list[str] = []
    onehotK_list: list[np.ndarray] = []
    onehotK_pri_list: list[np.ndarray] = []

    for i in range(len(seq) - 21):
        pRNA = seq[i: i + 22]
        gRNA = pRNA.translate(trans2)[::-1]
        # gRNA's 5'end should be unpaired
        if gRNA[-1] in ["A", "T"]:
            pRNA = "C" + pRNA[1:]
        elif gRNA[-1] in ["C", "G"]:
            pRNA = "A" + pRNA[1:]
        if gRNA in seqK:
            continue
        seqK.append(gRNA)

        var = list(map(int, list(gRNA.translate(trans))))
        var_onehot = np.eye(4)[var]
        onehotK_list.append(var_onehot)

        # pri-miR-30 templates
        seq_pri = (
            "GGTATATTGCTGTTGACAGTGAGCG"
            + pRNA
            + "TAGTGAAGCCACAGATGTA"
            + gRNA
            + "TGCCTACTGCCTCGGAATTCAAGGG"
        )
        priK.append(seq_pri)
        var_pri = list(map(int, list(seq_pri.translate(trans))))
        tempIn1 = np.delete(np.eye(5)[var_pri[:56]], -1, 1)
        tempIn2 = np.delete(np.eye(5)[var_pri[-56:][::-1]], -1, 1)
        tempIn3 = []
        for j in range(len(tempIn1)):
            seq1 = seq_pri[:56][j]
            seq2 = seq_pri[-56:][::-1][j]
            tempIn3.append(pair(seq1, seq2, j))

        seqIn = np.append(
            np.append(tempIn1, tempIn2, axis=1),
            np.asarray(tempIn3).reshape(len(tempIn3), 1),
            axis=1,
        )
        onehotK_pri_list.append(seqIn)

    onehotK = np.asarray(onehotK_list).reshape(-1, 22, 4, 1)
    onehotK_pri = np.asarray(onehotK_pri_list).reshape(-1, 56, 9, 1)
    return seqK, priK, onehotK, onehotK_pri


def get_Annotation(annoF: str) -> dict[str, list[str]]:
    # grep gene annotations
    # gencode v36 annotation
    annoDic: dict[str, list[str]] = {}
    with open(annoF, encoding="utf-8") as f:
        lines = f.readlines()
    for raw_line in lines:
        if raw_line.startswith("#"):
            continue
        fields = raw_line.strip().split("\t")
        line_type = fields[2]
        if line_type != "transcript":
            continue

        info_line = fields[8].strip(";").split("; ")
        symbol = list(
            filter(lambda x: x.split(" ")[0] == "gene_name", info_line)
        )[0].split('"')[1]
        txnID = list(
            filter(lambda x: x.split(" ")[0] == "transcript_id", info_line)
        )[0].split('"')[1]
        if txnID in ["ENST00000615113.1"]:
            continue
        if symbol not in annoDic:
            annoDic[symbol] = []
        annoDic[symbol].append(txnID)
    return annoDic


def get_Sequence(
    inF: str,
    region: str,
    annoDic: dict[str, list[str]],
) -> tuple[dict[str, str], dict[str, list[str]]]:
    # grep gene sequence
    # gencode v36 annotation
    seqDic: dict[str, str] = {}
    pairDic: dict[str, list[str]] = {}
    with open(inF, encoding="utf-8") as f:
        fasta_content = f.read()
    chunks = fasta_content.split(">")
    for chunk in chunks[1:]:
        lines = chunk.split("\n")

        info_line = lines[0].split("|")

        txnID = info_line[0]
        symbol = info_line[5]
        if txnID not in annoDic[symbol]:
            continue
        cdsCoord = list(filter(lambda x: x.split(":")[0] == "CDS", info_line))
        if len(cdsCoord) != 1:
            print(info_line)
            continue
        cdsStart = int(cdsCoord[0].split(":")[1].split("-")[0])
        cdsEnd = int(cdsCoord[0].split(":")[1].split("-")[1])

        seq = "".join(lines[1:])
        if region == "CDS":
            seqDic[txnID] = seq[cdsStart - 1: cdsEnd]
        elif region == "UTR":
            seqDic[txnID] = seq[cdsEnd:]
        if symbol not in pairDic:
            pairDic[symbol] = []
        pairDic[symbol].append(txnID)
    return seqDic, pairDic
