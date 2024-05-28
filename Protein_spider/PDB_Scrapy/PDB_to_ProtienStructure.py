from pymongo import MongoClient #pymongo 版本3.12.1
#from gridfs import GridFS
#import psycopg2

import pandas as pd
import re
import tqdm
#import requests
import urllib.request as request
from bs4 import BeautifulSoup
#import json

# 建立mongodb连接
mongoClient = MongoClient('mongodb://localhost:27017/')
#mongoClient = MongoClient('mongodb://172.10.10.9:27017/')#这个是我们要访问的mongo数据库的URL服务器地址
dtmr_dev = mongoClient.get_database('dtmr_dev')#访问的数据库名字为dtmr_dev
#dtmr_dev.authenticate('dtmr_dev', 'dtmr_dev')#数据库dtmr_dev的密码也是dtmr_dev
dev = dtmr_dev.get_collection('ProteinStructure')#collection 在mongodb里面相当于SQL中的表table，这句话相当于访问dtrm_dev数据库中的proteinStructure表
# gfs_pdb = GridFS(dtmr_dev, collection='PDB_files')
# gfs_seq = GridFS(dtmr_dev, collection='protein_seq_files')

# GridFS 用于存储和恢复那些超过16M（BSON文件限制）的文件(如：图片、音频、视频等)。
# GridFS 也是文件存储的一种方式，但是它是存储在MonoDB的集合中。
# GridFS 可以更好的存储大于16M的文件。
# GridFS 会将大文件对象分割成多个小的chunk(文件片段),一般为256k/个,每个chunk将作为MongoDB的一个文档(document)被存储在chunks集合中。
# GridFS 用两个集合来存储一个文件：fs.files与fs.chunks。


length = len(pd.read_csv("PDB_IDS_FOR_FILES.csv").keyword.values)

code = dev.count_documents({})
print(code, "of", length, ",", round(code / length * 100, 2), "% done")
pdb_ids = pd.read_csv("PDB_IDS_FOR_FILES.csv").keyword.values[code:]
count = 0
for pdb_id in tqdm.tqdm(pdb_ids):
    count += 1
    record = {}#定义一个集合
    html_url = "https://www.rcsb.org/structure/" + pdb_id
    response = request.urlopen(html_url)
    html = response.read()
    print('html:', html)
    soup = BeautifulSoup(html, 'html.parser')#beautifulsoup(解析内容，解析器)
    #beautifulsoup是一个解析器，可以特定的解析出内容，省去了我们编写正则表达式进行正则查询的麻烦。这里指定Beautiful的解析器为“html.parser”
    # 编号，例如PROT000001
    code = code + 1
    record['code'] = 'PROT' + '0' * (6 - len(str(code))) + str(code)

    record['PDBID'] = pdb_id

    LigandsMainTable = soup.find("table", id="LigandsMainTable")

    '''
    soup = BeautifulSoup(html, 'html.parser')，可见soup是已经解析完了的Html代码。
    所以LigandsMainTable = soup.find("table", id="LigandsMainTable")，这句代码就是在解析出的Html代码中找table，且id="LigandsMainTable"
    print(LigandsMainTable)后其中一条结果：
    <table class="table table-bordered table-condensed" id="LigandsMainTable"><tr class="active"><th colspan="6">Ligands <span class="badge">3 Unique</span></th></tr><tr><th class="col-sm-2 col-lg-2 pad-by-five">ID</th><th class="col-sm-1 col-lg-1 pad-by-five">Chains<span aria-hidden="true" class="glyphicon glyphicon-info-sign hidden-xs hidden-sm hidden-md" data-original-title="If the two PDB chain IDs (label_asym_id; assigned by the PDB) and auth_asym_id (selected by the author) do not coincide, the chain ID is displayed as “label_asym_id [auth auth_asym_id]”" data-placement="top" data-toggle="tooltip"></span></th><th class="col-sm-3 col-lg-4 pad-by-five">Name / Formula / InChI Key</th><th class="col-sm-3 col-lg-3 pad-by-five">2D Diagram</th><th class="col-sm-2 col-lg-2 pad-by-five">3D Interactions</th></tr><tbody><a name="chem-comp-FFO"></a><tr id="ligand_row_FFO"><td><a href="/ligand/FFO">FFO</a><br/><a class="hidden-print querySearchLink" href="/search?q=rcsb_chem_comp_container_identifiers.comp_id:FFO">Query on FFO</a><br/><hr/><a class="btn btn-default btn-xs hidden-print" href="https://files.rcsb.org/ligands/download/FFO.cif">Download Ideal Coordinates CCD File <span class="glyphicon glyphicon-download"></span></a><br/><div style="display: flow-root; margin-bottom: 5px;"><div class="btn-group" role="group" style="position: absolute"><button aria-expanded="false" class="btn btn-xs btn-default dropdown-toggle" data-toggle="dropdown" id="dropdownMenuDisplayCordinateFiles" role="group" type="button">Download Instance Coordinates <span class="caret"></span></button><ul aria-labelledby="dropdownMenuDisplayCordinateFiles" class="dropdown-menu" role="menu"><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=402&amp;label_asym_id=M&amp;encoding=sdf&amp;filename=6qyq_M_FFO.sdf">SDF format, chain M [auth B]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=402&amp;label_asym_id=P&amp;encoding=sdf&amp;filename=6qyq_P_FFO.sdf">SDF format, chain P [auth D]</a></li><li class="divider"></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=402&amp;label_asym_id=M&amp;encoding=mol2&amp;filename=6qyq_M_FFO.mol2">MOL2 format, chain M [auth B]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=402&amp;label_asym_id=P&amp;encoding=mol2&amp;filename=6qyq_P_FFO.mol2">MOL2 format, chain P [auth D]</a></li></ul></div></div></td><td>M [auth B],<br>P [auth D]</br></td><td><strong>N-[4-({[(6S)-2-amino-5-formyl-4-oxo-3,4,5,6,7,8-hexahydropteridin-6-yl]methyl}amino)benzoyl]-L-glutamic acid</strong><br/>C<sub>20</sub> H<sub>23</sub> N<sub>7</sub> O<sub>7</sub><br/>VVIAGPKUTFNRDU-STQMWFEESA-N</td><td><a data-title="6QYQ: Ligand FFO" data-toggle="lightbox" href="https://cdn.rcsb.org/images/ccd/labeled/F/FFO.svg"><img src="https://cdn.rcsb.org/images/ccd/unlabeled/F/FFO.svg" style="width:116px; margin: 0 25%;"/></a></td><td><a class="btn btn-primary btn-sm" href="/3d-view/6QYQ?preset=ligandInteraction&amp;label_asym_id=M" role="button"><i class="fa fa-cube"> </i>Ligand Interaction</a></td></tr><a name="chem-comp-SO4"></a><tr id="ligand_row_SO4"><td><a href="/ligand/SO4">SO4</a><br/><a class="hidden-print querySearchLink" href="/search?q=rcsb_chem_comp_container_identifiers.comp_id:SO4">Query on SO4</a><br/><hr/><a class="btn btn-default btn-xs hidden-print" href="https://files.rcsb.org/ligands/download/SO4.cif">Download Ideal Coordinates CCD File <span class="glyphicon glyphicon-download"></span></a><br/><div style="display: flow-root; margin-bottom: 5px;"><div class="btn-group" role="group" style="position: absolute"><button aria-expanded="false" class="btn btn-xs btn-default dropdown-toggle" data-toggle="dropdown" id="dropdownMenuDisplayCordinateFiles" role="group" type="button">Download Instance Coordinates <span class="caret"></span></button><ul aria-labelledby="dropdownMenuDisplayCordinateFiles" class="dropdown-menu" role="menu"><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=401&amp;label_asym_id=E&amp;encoding=sdf&amp;filename=6qyq_E_SO4.sdf">SDF format, chain E [auth A]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=402&amp;label_asym_id=F&amp;encoding=sdf&amp;filename=6qyq_F_SO4.sdf">SDF format, chain F [auth A]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=403&amp;label_asym_id=G&amp;encoding=sdf&amp;filename=6qyq_G_SO4.sdf">SDF format, chain G [auth A]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=401&amp;label_asym_id=I&amp;encoding=sdf&amp;filename=6qyq_I_SO4.sdf">SDF format, chain I [auth C]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=401&amp;label_asym_id=L&amp;encoding=sdf&amp;filename=6qyq_L_SO4.sdf">SDF format, chain L [auth B]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=401&amp;label_asym_id=O&amp;encoding=sdf&amp;filename=6qyq_O_SO4.sdf">SDF format, chain O [auth D]</a></li><li class="divider"></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=401&amp;label_asym_id=E&amp;encoding=mol2&amp;filename=6qyq_E_SO4.mol2">MOL2 format, chain E [auth A]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=402&amp;label_asym_id=F&amp;encoding=mol2&amp;filename=6qyq_F_SO4.mol2">MOL2 format, chain F [auth A]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=403&amp;label_asym_id=G&amp;encoding=mol2&amp;filename=6qyq_G_SO4.mol2">MOL2 format, chain G [auth A]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=401&amp;label_asym_id=I&amp;encoding=mol2&amp;filename=6qyq_I_SO4.mol2">MOL2 format, chain I [auth C]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=401&amp;label_asym_id=L&amp;encoding=mol2&amp;filename=6qyq_L_SO4.mol2">MOL2 format, chain L [auth B]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=401&amp;label_asym_id=O&amp;encoding=mol2&amp;filename=6qyq_O_SO4.mol2">MOL2 format, chain O [auth D]</a></li></ul></div></div></td><td><div class="hidden-print" id="asymPart_5">E [auth A],<br>F [auth A],<br>G [auth A],<br>I [auth C],<br>L [auth B],<br><span class="glyphicon glyphicon-plus-sign" id="full_asym_show_5"></span></br></br></br></br></br></div><div class="hidden-print hide" id="asymFull_5">E [auth A],<br>F [auth A],<br>G [auth A],<br>I [auth C],<br>L [auth B],<br>O [auth D]<br/><button class="btn btn-default btn-xs hidden-print" id="full_asym_hide_5"><span class="glyphicon glyphicon-minus-sign"></span> Less</button></br></br></br></br></br></div></td><td><strong>SULFATE ION</strong><br/>O<sub>4</sub> S<br/>QAOWNCQODCNURD-UHFFFAOYSA-L</td><td><a data-title="6QYQ: Ligand SO4" data-toggle="lightbox" href="https://cdn.rcsb.org/images/ccd/labeled/S/SO4.svg"><img src="https://cdn.rcsb.org/images/ccd/unlabeled/S/SO4.svg" style="width:116px; margin: 0 25%;"/></a></td><td><a class="btn btn-primary btn-sm" href="/3d-view/6QYQ?preset=ligandInteraction&amp;label_asym_id=E" role="button"><i class="fa fa-cube"> </i>Ligand Interaction</a></td></tr><a name="chem-comp-CL"></a><tr id="ligand_row_CL"><td><a href="/ligand/CL">CL</a><br/><a class="hidden-print querySearchLink" href="/search?q=rcsb_chem_comp_container_identifiers.comp_id:CL">Query on CL</a><br/><hr/><a class="btn btn-default btn-xs hidden-print" href="https://files.rcsb.org/ligands/download/CL.cif">Download Ideal Coordinates CCD File <span class="glyphicon glyphicon-download"></span></a><br/><div style="display: flow-root; margin-bottom: 5px;"><div class="btn-group" role="group" style="position: absolute"><button aria-expanded="false" class="btn btn-xs btn-default dropdown-toggle" data-toggle="dropdown" id="dropdownMenuDisplayCordinateFiles" role="group" type="button">Download Instance Coordinates <span class="caret"></span></button><ul aria-labelledby="dropdownMenuDisplayCordinateFiles" class="dropdown-menu" role="menu"><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=404&amp;label_asym_id=H&amp;encoding=sdf&amp;filename=6qyq_H_CL.sdf">SDF format, chain H [auth A]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=402&amp;label_asym_id=J&amp;encoding=sdf&amp;filename=6qyq_J_CL.sdf">SDF format, chain J [auth C]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=403&amp;label_asym_id=K&amp;encoding=sdf&amp;filename=6qyq_K_CL.sdf">SDF format, chain K [auth C]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=403&amp;label_asym_id=N&amp;encoding=sdf&amp;filename=6qyq_N_CL.sdf">SDF format, chain N [auth B]</a></li><li class="divider"></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=404&amp;label_asym_id=H&amp;encoding=mol2&amp;filename=6qyq_H_CL.mol2">MOL2 format, chain H [auth A]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=402&amp;label_asym_id=J&amp;encoding=mol2&amp;filename=6qyq_J_CL.mol2">MOL2 format, chain J [auth C]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=403&amp;label_asym_id=K&amp;encoding=mol2&amp;filename=6qyq_K_CL.mol2">MOL2 format, chain K [auth C]</a></li><li><a href="https://models.rcsb.org/v1/6qyq/ligand?auth_seq_id=403&amp;label_asym_id=N&amp;encoding=mol2&amp;filename=6qyq_N_CL.mol2">MOL2 format, chain N [auth B]</a></li></ul></div></div></td><td>H [auth A],<br>J [auth C],<br>K [auth C],<br>N [auth B]</br></br></br></td><td><strong>CHLORIDE ION</strong><br/>Cl<br/>VEXZGXHMUGYJMC-UHFFFAOYSA-M</td><td><a data-title="6QYQ: Ligand CL" data-toggle="lightbox" href="https://cdn.rcsb.org/images/ccd/labeled/C/CL.svg"><img src="https://cdn.rcsb.org/images/ccd/unlabeled/C/CL.svg" style="width:116px; margin: 0 25%;"/></a></td><td><a class="btn btn-primary btn-sm" href="/3d-view/6QYQ?preset=ligandInteraction&amp;label_asym_id=H" role="button"><i class="fa fa-cube"> </i>Ligand Interaction</a></td></tr></tbody></table>
    由<table class="table table-bordered table-condensed" id="LigandsMainTable">可以看出，我们已经找到了。
    
    '''

    #find     find_all

    '''
    find(name,attrs,recursive,text)函数通过参数来找出对应的标签，但只返回第一个符合的结果，find_all(...)则是返回所有符合条件的结果。
    其中name为通过标签名来筛选标签。
    attrs为通过属性键值来筛选标签 如attrs={属性名:值}
    text为根据指定文本内容来筛选标签
    recursive为指定筛选是否为递归，当为False时，不会在子结点的后代结点中查找，只会查找子结点。
    '''
    if LigandsMainTable == None:
        record["ligands"] = None
    else:
        ligand_rows = LigandsMainTable.find_all('tr', {'id': re.compile('ligand_row_*')})
        '''
        print(ligand_rows)
        [<tr id="ligand_row_ADZ"><td><a href="/ligand/ADZ">ADZ</a><br/><a class="hidden-print querySearchLink" href="/search?q=rcsb_chem_comp_container_identifiers.comp_id:ADZ">Query on ADZ</a><br/><hr/><a class="btn btn-default btn-xs hidden-print" href="https://files.rcsb.org/ligands/download/ADZ.cif">Download Ideal Coordinates CCD File <span class="glyphicon glyphicon-download"></span></a><br/><div style="display: flow-root; margin-bottom: 5px;"><div class="btn-group" role="group" style="position: absolute"><button aria-expanded="false" class="btn btn-xs btn-default dropdown-toggle" data-toggle="dropdown" id="dropdownMenuDisplayCordinateFiles" role="group" type="button">Download Instance Coordinates <span class="caret"></span></button><ul aria-labelledby="dropdownMenuDisplayCordinateFiles" class="dropdown-menu" role="menu"><li><a href="https://models.rcsb.org/v1/1o9u/ligand?auth_seq_id=1386&amp;label_asym_id=C&amp;encoding=sdf&amp;filename=1o9u_C_ADZ.sdf">SDF format, chain C [auth A]</a></li><li class="divider"></li><li><a href="https://models.rcsb.org/v1/1o9u/ligand?auth_seq_id=1386&amp;label_asym_id=C&amp;encoding=mol2&amp;filename=1o9u_C_ADZ.mol2">MOL2 format, chain C [auth A]</a></li></ul></div></div></td><td>C [auth A]</td><td><strong>9-METHYL-9H-PURIN-6-AMINE</strong><br/>C<sub>6</sub> H<sub>7</sub> N<sub>5</sub><br/>WRXCXOUDSPTXNX-UHFFFAOYSA-N</td><td><a data-title="1O9U: Ligand ADZ" data-toggle="lightbox" href="https://cdn.rcsb.org/images/ccd/labeled/A/ADZ.svg"><img src="https://cdn.rcsb.org/images/ccd/unlabeled/A/ADZ.svg" style="width:116px; margin: 0 25%;"/></a></td><td><a class="btn btn-primary btn-sm" href="/3d-view/1O9U?preset=ligandInteraction&amp;label_asym_id=C" role="button"><i class="fa fa-cube"> </i>Ligand Interaction</a></td></tr>]
        
        '''
        ligands = []
        for row in ligand_rows:
            col = row.find_all("td")[2]
            '''注意使用和区分集合和字典和列表和元组'''
            ligand = {}
            ligand["Name"] = col.find("strong").text
            ligand["InChI Key"] = str(col.decode_contents()).split("<br/>")[-1]
            ligands.append(ligand)
        record["ligands"] = ligands
    if count == 2:
        break

    # r = requests.get("https://www.rcsb.org/fasta/entry/" + pdb_id, allow_redirects=True)
    # if r.status_code != 200:
    #     record["sequence"] = None
    # else:
    #     record["sequence"] = gfs_seq.put(r.content, filename=pdb_id+".fasta", type="fasta", keyword=pdb_id)

    # r = requests.get("https://files.rcsb.org/download/" + pdb_id + ".pdb", allow_redirects=True)
    # if r.status_code != 200:
    #     record["structure3d"] = None
    # else:
    #     record["structure3d"] = gfs_pdb.put(r.content, filename=pdb_id+".pdb", type="pdb", keyword=pdb_id)

#     records.append(record)
result = dev.insert_one(record)