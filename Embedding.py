import pandas as pd
import gc
from random import choice, shuffle
import random
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os,sys
from multiprocessing import Process, Queue, Manager
from scipy import stats
from sklearn.metrics import roc_curve, auc

vnum = {'cnv':5,'sv':5,'MutaC':20,'ncv':10}
pnum = 10
gnum = 10

def inputwgs(datapath):

    sample_cnv = dict()
    var_gene = dict()

    cnv = pd.read_csv(datapath + 'cnv.csv', header=0)
    for j in list(cnv.columns)[1:]:
        sample_cnv['sample_'+j] = set()
    for i in cnv.index:
        for j in list(cnv.columns)[1:]:
            if cnv.loc[i,j]<-1:
                sample_cnv['sample_'+j].add('variant_cnv_' + str(i))
                var_gene['variant_cnv_' + str(i)]={'gene_'+cnv.loc[i, 'ID'].split('|')[0]}
            elif cnv.loc[i,j]>1:
                sample_cnv['sample_'+j].add('variant_cnv_' + str(i))
                var_gene['variant_cnv_' + str(i)]={'gene_'+cnv.loc[i, 'ID'].split('|')[0]}
    del cnv
    gc.collect()

    sv = pd.read_csv(datapath + 'sv_full.csv', header=0)
    sample_sv = dict()
    for j in set(sv['Sample_ID']):
        sample_sv['sample_'+j] = set()
    for i in sv.index:
        if not pd.isnull(sv.loc[i, 'Gene.name']):
            var_gene['variant_sv_' + str(i)] = {'gene_'+x for x in set(str(sv.loc[i, 'Gene.name']).split('/'))}
            sample_sv['sample_' + sv.loc[i, 'Sample_ID']].add('variant_sv_' + str(i))
    del sv
    gc.collect()

    muta_coding = pd.read_csv(datapath + 'PDAC_coding_mutation.CSV', header=0)
    sample_coding_muta = dict()
    samples = list(set(muta_coding['Sample_ID'].values))
    genes = list(set(muta_coding['Gene.refGene'].values))
    gene_sample_num = pd.DataFrame(0, index=genes, columns=samples)  # counts of samples mutation num in the gene
    for j in muta_coding.index:
        sample = muta_coding.loc[j, 'Sample_ID']
        if not 'sample_' + sample in sample_coding_muta:
            sample_coding_muta['sample_' + sample] = set()
        mutaID = muta_coding.loc[j, 'Chr'] + '_' + str(muta_coding.loc[j, 'Start']) + str(muta_coding.loc[j, 'Alt'])
        sample_coding_muta['sample_' + sample].add('variant_MutaC_' + mutaID)
        var_gene['variant_MutaC_' + mutaID] = {'gene_' + muta_coding.loc[j, 'Gene.refGene']}
        gene_sample_num.loc[muta_coding.loc[j, 'Gene.refGene'], sample] += 1
    del muta_coding
    gc.collect()
    gene_sample_num.to_csv(source_path+'./gene_sample_num_coding.csv')

    muta_ncv = pd.read_csv(datapath + './ncv2peak.csv')
    peak2gene = pd.read_csv(datapath + './peak2gene.csv')
    samples = list(s for slist in muta_ncv['samples'] for s in slist.split('|'))
    genes = list(set(peak2gene['target gene']))
    gene_sample_num = pd.DataFrame(0, index=genes, columns=samples)  # counts of samples mutation num in the gene
    peak_gene = {'peak_'+p:set() for p in set(peak2gene['peak'].values)}
    for i in peak2gene.index:
        peak_gene['peak_'+peak2gene.loc[i,'peak']].add(peak2gene.loc[i,'target gene'])
    sample_ncv = {'sample_'+s:set() for s in samples}
    for i in muta_ncv.index:
        samples = muta_ncv.loc[i, 'samples'].split('|')
        for sample in samples:
            sample_ncv['sample_' + sample].add('variant_' + muta_ncv.loc[i, 'id'])
        peaks = muta_ncv.loc[i,'peaks'].split('|')
        var_gene['variant_' + muta_ncv.loc[i, 'id']]=set(['peak_'+p for p in peaks])
        genes = {}
    del muta_ncv
    gc.collect()

    samples = set(sample_cnv.keys()) | set(sample_sv.keys()) | set(sample_coding_muta.keys()) \
              | set(sample_ncv.keys())
    sample_wgs = dict()

    for sample in samples:
        sample_wgs[sample] = dict()
        if not sample in sample_cnv:
            sample_cnv[sample] = set()
        sample_wgs[sample]['cnv'] = sample_cnv[sample]
        if not sample in sample_sv:
            sample_sv[sample] = set()
        sample_wgs[sample]['sv'] = sample_sv[sample]
        if not sample in sample_coding_muta:
            sample_coding_muta[sample] = set()
        sample_wgs[sample]['MutaC'] = sample_coding_muta[sample]
        if not sample in sample_ncv:
            sample_ncv[sample] = set()
        sample_wgs[sample]['ncv'] = sample_ncv[sample]

    return sample_wgs, var_gene


'''
sample_wgs: {sample:{'cnv':{'cnv_ID'}, 
                     'sv':{'sv_svID'},
                     'MutaC':{'MutaC_ID'},
                     'ncv':{'ncv_ncvID'}}
                     }
'''


def importpathways(datapath,geneset):
    gene_pathway = dict()
    pathways_relation = dict()
    pathway_gene = dict()
    node = dict()

    node['pathway']=set()
    with open(datapath + './ReactomePathways.gmt', 'r') as f:
        for line in f:
            l = line[:-1].split('\t')
            for gene in set(l[2:])&geneset:
                if not 'gene_'+gene in gene_pathway:
                    gene_pathway['gene_'+gene] = set()
                gene_pathway['gene_'+gene].add('pathway_'+l[1])
                node['pathway'].add('pathway_'+l[1])
            pathway_gene['pathway_'+l[1]] = {'gene_'+x for x in l[2:]}
    #print(gene_pathway)
    #print(len(node['pathway']))
    #print(len(pathway_gene))
    with open(datapath + './ReactomePathwaysRelation.txt', 'r') as f:
        for line in f:
            l = line[:-1].split('\t')
            if 'pathway_'+l[1] in pathway_gene and 'pathway_'+l[0] in pathway_gene:
                if not 'pathway_'+l[1] in pathways_relation:
                    pathways_relation['pathway_'+l[1]] = set()
                #if not l[0] in pathways_relation:
                #    pathways_relation[l[0]] = set()
                pathways_relation['pathway_'+l[1]].add('pathway_'+l[0])
                #node['pathway'].add('pathway_'+l[1])
                #node['pathway'].add('pathway_'+l[0])
    node['gene'] = set(gene_pathway.keys())
    return gene_pathway, pathways_relation, pathway_gene, node

'''
gene_pathway: {gene:{pathways}}
pathways_relation: {pathway:{pathways}}
'''

def getppi(ppi_path, geneset):
    gene_gene = dict()
    with open(ppi_path,'r') as f:
        for line in f:
            g1 = line[:-1].split('\t')[0]
            g2 = line[:-1].split('\t')[1]
            if 'gene_'+g1 not in geneset or 'gene_'+g2 not in geneset:
                continue
            if 'gene_'+g1 not in gene_gene:
                gene_gene['gene_'+g1] = set()
            if 'gene_'+g2 not in gene_gene:
                gene_gene['gene_'+g2] = set()
            gene_gene['gene_'+g1].add('gene_'+g2)
            gene_gene['gene_'+g2].add('gene_'+g1)
    return gene_gene

def getnodes(sample_wgs, var_gene):
    node = dict()
    node['sample'] = set(sample_wgs.keys())
    node['variant'] = set(var_gene.keys())
    print(len(node['variant']))
    return node

def getpathmeans():
    means = dict()
    with open('./pathways/ReactomePathways.gmt','r') as f:
        for line in f:
            l = line.split('\t')
            means[l[1]]=l[0]
    return means

def inverse(sample_wgs,var_gene):
    gene_wgs = dict()
    wgs_sample = dict()
    for sample in sample_wgs.keys():
        for vtype in sample_wgs[sample].keys():
            for variant in sample_wgs[sample][vtype]:
                if not variant in wgs_sample:
                    wgs_sample[variant] = set()
                wgs_sample[variant].add(sample)
                for gene in var_gene[variant]:
                    if not gene in gene_wgs:
                        gene_wgs[gene] = set()
                        #gene_wgs[gene] = list()
                    gene_wgs[gene].add(variant)
                    #gene_wgs[gene] +=[variant]*vnum[vtype]
    return gene_wgs, wgs_sample

#return gene_wgs, wgs_sample, pathway_gene (in dict)
def getVGPPP(v,var_gene,gene_pathway,pathways_relation):
    walktemp = [v]
    if len(var_gene[v]) != 0:
        gene = choice(list(var_gene[v]))
    else:
        return walktemp

    walktemp.append(gene)
    if gene in gene_pathway.keys():
        p = choice(list(gene_pathway[gene]))
        walktemp.append(p)
        while p in pathways_relation.keys():
            p = choice(list(pathways_relation[p]))
            walktemp.append(p)
    return walktemp

def choicev(s, vtype,sample_wgs):
    if vtype in {'MutaC', 'cnv'}:
        return choice(list(sample_wgs[s]['MutaC'] | sample_wgs[s]['cnv']))
    else:
        return choice(list(sample_wgs[s][vtype]))

def walkGVSVG(walksgene,gene_wgs,wgs_sample,sample_wgs,var_gene,walknum,geneset,label):
    twalks = list()
    c = 0
    for gene in geneset:  # GVSVG
        gvlist = list()
        for v1 in list(gene_wgs[gene]):
            gvlist = gvlist + [v1] * vnum[v1.split('_')[1]]
        #gvlist = gene_wgs[gene]
        for v1 in gvlist:
            tempwalk = [gene, v1]
            v = v1
            temp = 0
            while temp + len(tempwalk) < len(wgs_sample[v])*len(gene_wgs[gene])*4: # len(wgs_sample[v])*len(gene_wgs[gene])*L:
                s = choice(list(wgs_sample[v]))
                tempwalk.append(s)

                if GVSVGUseTraits and len(sample_trait[s])!=0:
                    tr = choice(list(sample_trait[s]))
                    tempwalk.append(tr)
                    s = choice(list(trait_sample[tr]))
                    tempwalk.append(s)
                    if len(list(sample_wgs[s][v.split('_')[1]])) == 0:
                        temp += len(tempwalk)
                        walksgene.append(tempwalk)
                        s = choice(list(wgs_sample[v1]))
                        tempwalk = [gene,v1,s]
                        while len(list(sample_wgs[s][v.split('_')[1]])) == 0:
                            s = choice(list(trait_sample[tr]))
                        tempwalk.append(s)
                variants = sample_wgs[s]['cnv'] | \
                           sample_wgs[s]['sv'] | \
                           sample_wgs[s]['MutaC'] | \
                           sample_wgs[s]['ncv']
                #v = choice(list(variants))
                #v = choice(list(sample_wgs[s][vtype]))
                v = choicev(s, v.split('_')[1],sample_wgs)
                tempwalk.append(v)
                g = choice(list(var_gene[v]))
                tempwalk.append(g)
                v = choice(list(gene_wgs[g]))
                tempwalk.append(v)
                # print(len(tempwalk))
                # print([s1,v1,gene,v2,s2])
            v = choice(list(gene_wgs[gene]))
            s = choice(list(wgs_sample[v]))
            walksgene.append([s,v]+tempwalk)
            # print(gene,v1)
        c += 1
        if c % 10 == 0:
            print('\r%s SVGVS %d in %d'%(label, c,len(geneset)), end='')
    #walksgene += twalks
    return walksgene

def walkPGVSVGP(walkspath,pathway_gene,gene_wgs,wgs_sample,sample_wgs,var_gene,gene_pathway,walknum,pathwayset,label):
    #twalks = list()
    temp = 0
    for pathway in pathwayset:  # PGVSVGP
        for g1 in list(pathway_gene[pathway]):
            if g1 in gene_wgs.keys():
                tempwalk = [pathway, g1]
                templen = 0
                g = g1
                while templen + len(tempwalk) < len(gene_wgs[g1])*L:
                    v = choice(list(gene_wgs[g]))
                    tempwalk.append(v)
                    s = choice(list(wgs_sample[v]))
                    tempwalk.append(s)
                    variants = sample_wgs[s]['cnv'] | sample_wgs[s]['sv'] | \
                               sample_wgs[s]['MutaC'] | sample_wgs[s]['ncv']
                    #v = choice(list(variants))
                    # v = choice(list(sample_wgs[s][vtype]))
                    v = choicev(s, v.split('_')[1], sample_wgs)

                    tempwalk.append(v)
                    if len(var_gene[v] & set(gene_pathway.keys())) == 0:
                        g = choice(list(var_gene[v]))
                        tempwalk.append(g)
                        walkspath.append(tempwalk)
                        templen += len(tempwalk)
                        tempwalk = [pathway, g1]
                        g = g1
                    else:
                        g = choice(list(var_gene[v] & set(gene_pathway.keys())))
                        tempwalk.append(g)
                        p = choice(list(gene_pathway[g]))
                        tempwalk.append(p)
                        if len(pathway_gene[p] & set(gene_wgs.keys())) == 0:
                            g = choice(list(pathway_gene[p]))
                            tempwalk.append(g)
                            walkspath.append(tempwalk)
                            templen += len(tempwalk)
                            tempwalk = [pathway, g1]
                            g = g1
                        else:
                            g = choice(list(pathway_gene[p] & set(gene_wgs.keys())))
                            tempwalk.append(g)
                walkspath.append(tempwalk)
        temp += 1
        if temp % 20 == 0:
            print( '\r%s SVGPGVS %d' %(label,temp), end='')
    #walkspath += twalks
    return walkspath

def walkGGG(walksgene,gene_gene,walknum,geneset,label):
    #twalks = list()
    c = 0
    for gene in geneset:
        temp = 0
        while temp < len(gene_gene[gene])*L:
            tempwalk = [gene]
            temp += 1
            ng = choice(list(gene_gene[gene]))
            while ng not in tempwalk and temp < len(gene_gene[gene]):
                tempwalk.append(ng)
                temp += 1
                ng = choice(list(gene_gene[ng]))
            walksgene.append(tempwalk)
        c+=1
        if c%100==0:
            print('\r%s GGG %d'%(label,c),end='')
    return walksgene

def walkSVGGVS(walksgene,gene_wgs,wgs_sample,sample_wgs,var_gene,gene_gene,geneset,label):
    #twalks = list()
    c=0
    for gene in geneset:
        if len(gene_gene[gene]&set(gene_wgs.keys()))==0:
            continue
        temp = 0
        tempwalk = [gene]
        g = gene
        while temp + len(tempwalk) < len(gene_gene[gene])*L:
            g = choice(list(gene_gene[g]&set(gene_wgs.keys())))
            tempwalk.append(g)
            v = choice(list(gene_wgs[g]))
            tempwalk.append(v)
            s = choice(list(wgs_sample[v]))
            tempwalk.append(s)
            variants = sample_wgs[s]['cnv'] | sample_wgs[s]['sv'] | \
                        sample_wgs[s]['MutaC'] | sample_wgs[s]['ncv']
            #v = choice(list(variants))
            # v = choice(list(sample_wgs[s][vtype]))
            v = choicev(s, v.split('_')[1], sample_wgs)

            tempwalk.append(v)
            if len(var_gene[v] & set(gene_gene.keys())) == 0:
                g = choice(list(var_gene[v]))
                tempwalk.append(g)
                temp += len(tempwalk)
                walksgene.append(tempwalk)
                tempwalk = [gene]
                g = gene
            else:
                g = choice(list(var_gene[v] & set(gene_gene.keys())))
                tempwalk.append(g)
                if len(gene_gene[g] & set(gene_wgs.keys())) == 0:
                    temp += len(tempwalk)
                    walksgene.append(tempwalk)
                    tempwalk = [gene]
                    g = gene
        walksgene.append(tempwalk)
        c+=1
        if c%100==0:
            print('\r%s SVGGVS %d'%(label,c),end='')
    return walksgene

def walkGGPGG(walksgene,gene_gene,gene_pathway,pathway_gene,geneset,label):
    #twalks = []
    c=0
    for gene in geneset:
        temp = 0
        #print(gene,len(gene_gene[gene]))
        if len(gene_gene[gene] & set(gene_pathway.keys())) == 0:
            continue
        while temp < len(gene_gene[gene])*L:
            g = choice(list(gene_gene[gene] & set(gene_pathway.keys())))
            tempwalk = [gene,g]
            while (g not in tempwalk) and (temp+len(tempwalk) < len(gene_gene[gene])*L):
                p = choice(list(gene_pathway[g]))
                tempwalk.append(p)
                if len(pathway_gene[p]&set(gene_gene.keys()))==0:
                    break
                else:
                    g = choice(pathway_gene[p]&set(gene_gene.keys()))
                    tempwalk.append(g)
                    if len(gene_gene[g] & set(gene_pathway.keys())) == 0:
                        break
                    g = choice(list(gene_gene[gene] & set(gene_pathway.keys())))
                    tempwalk.append(g)
            temp += len(tempwalk)
            walksgene.append(tempwalk)
        #print(twalks[-1])
        c+=1
        if c%100==0:
            print('\r%s GGPGG %d'%(label,c), end='')
    return walksgene


def walk(sample_wgs, var_gene, gene_pathway, pathways_relation, pathway_gene, walknum):
    gene_wgs, wgs_sample= inverse(sample_wgs, var_gene)
    global walks,sample_trait,trait_sample
    walks = list()
    sample_trait = dict()
    trait_sample = dict()
    if UseTraits:
        traits = pd.read_csv(Traits_path,index_col=1)
        for trait in {'F','M',
                      'IA','IB','IIA','IIB','III','IV',
                      'T1','T2','T3',
                      'N0','N1','N2','M0','M1',
                      'G12D','G12V','G12C','G12R','KRASWT','Q61H'}:
            trait_sample['trait_'+trait] = set()
        if Predict:
            global sample_test
            sample_with_trait = list({'sample_'+x for x in traits[traits[PredictTrait].notnull()].index} &
                                     set(sample_wgs.keys()))
            shuffle(sample_with_trait)
            sample_test = sample_with_trait[:int(len(sample_with_trait)*PredictRate)]
            sample_with_trait = {(x,PredictTrait) for x in sample_test}
        else:
            sample_with_trait = set()
        for sample in sample_wgs.keys():
            sample_trait[sample] = set()
            for t in ['Gender',
                      'Tumor_type',
                      'Tstage','Nstage','Mstage',
                      'KRAStype'
                      ]:
                if (not pd.isnull(traits.loc[sample[7:],t])) and ((sample,t) not in sample_with_trait):
                    sample_trait[sample].add('trait_'+traits.loc[sample[7:],t])
                    trait_sample['trait_'+traits.loc[sample[7:],t]].add(sample)
            #if len(sample_trait[sample]) == 0:
                #print(sample)
    t = 0
    if Drug:
        walks += walkDurg(gene_wgs, wgs_sample, sample_wgs, var_gene)
    print('SVGD done %d walks in all'%len(walks))
    for sample in sample_wgs.keys():  #GVSVGPPPP
        for vtype in sample_wgs[sample].keys():
            for v1 in sample_wgs[sample][vtype]:
                temp = 0
                vtgenenum = len(var_gene[v1])
                while temp < vnum[vtype] * vtgenenum :
                    walkv1 = getVGPPP(v1, var_gene, gene_pathway, pathways_relation)
                    s1 = sample
                    center = [sample]
                    if UseTraits and len(sample_trait[sample])!=0:
                        tr = choice(list(sample_trait[sample]))
                        s1 = choice(list(trait_sample[tr]))
                        center = [s1,tr,sample]
                    if len(sample_wgs[s1][vtype]) != 0:
                        v2 = choicev(s1, vtype, sample_wgs)

                        walkv2 = getVGPPP(v2, var_gene, gene_pathway, pathways_relation)
                        walkv2.reverse()
                    else:
                        walkv2 = []
                    walktemp = walkv2 + center + walkv1
                    temp += 1
                    walks.append(walktemp)
        t += 1
        print('\rSVGPPP %d'%t,end='')
    #model = Word2Vec(walks, size=dim, window=window, min_count=0, sg=0, workers=8, iter=15)
    print('\nSVGPP done %d walks'%len(walks))

    #walks = list()
    walksgene = Manager().list()
    N = 8
    geneset = []
    for i in range(N):
        geneset.append(set())
    for i in gene_wgs.keys():
        geneset[random.randint(0, N - 1)].add(i)
    processes = []
    for i in range(N):
        processes.append(
            Process(target=walkGVSVG,
                    args=(walksgene,gene_wgs,wgs_sample,sample_wgs,var_gene,walknum,geneset[i],'p'+str(i))))
    for i in range(N):
        processes[i].start()
    for i in range(N):
        processes[i].join()
    walks = walks + list(walksgene)

    #walks = walks + walkGVSVG([],gene_wgs,wgs_sample,sample_wgs,var_gene,walknum,gene_wgs.keys(),'p0')
    print('\nGVSVGdone %d walks'%len(walks))
    #model.train(walks,epochs=15,total_examples=model.corpus_count)

    #walks = list()
    walkspath = Manager().list()
    N = 8
    pathwayset = []
    for i in range(N):
        pathwayset.append(set())
    for i in pathway_gene.keys():
        pathwayset[random.randint(0, N - 1)].add(i)
    processes = []
    for i in range(N):
        processes.append(
            Process(target=walkPGVSVGP,
                    args=(walkspath,pathway_gene,gene_wgs,wgs_sample,sample_wgs,var_gene,gene_pathway,walknum,pathwayset[i],'p'+str(i))))
    for i in range(N):
        processes[i].start()
    for i in range(N):
        processes[i].join()
    walks = walks + list(walkspath)

    #walks = walks + walkPGVSVGP([],pathway_gene,gene_wgs,wgs_sample,sample_wgs,var_gene,gene_pathway,walknum,pathway_gene.keys(),'p0')
    print('\nPGVSVG done %d walks'%len(walks))
    #model.train(walks, epochs=15, total_examples=model.corpus_count)

    if ppi:

        #walks = list()
        gene_gene = getppi(ppi_path,set(gene_wgs.keys()))

        N = 8
        geneset = []
        for i in range(N):
            geneset.append(set())
        for i in gene_gene.keys():
            geneset[random.randint(0, N - 1)].add(i)

        walksgene = Manager().list()
        processes = []
        for i in range(N):
            processes.append(
                Process(target=walkGGG,
                        args=(walksgene, gene_gene, walknum, geneset[i], 'p'+str(i))))
        for i in range(N):
            processes[i].start()
        for i in range(N):
            processes[i].join()
        walks = walks + list(walksgene)
        print(len(walks))

        walksgene = Manager().list()
        processes = []
        for i in range(N):
            processes.append(
                Process(target=walkSVGGVS,
                        args=(walksgene, gene_wgs, wgs_sample, sample_wgs, var_gene, gene_gene, geneset[i], 'p' + str(i))))
        for i in range(N):
            processes[i].start()
        for i in range(N):
            processes[i].join()
        walks = walks + list(walksgene)
        print(len(walks))

        walksgene = Manager().list()
        processes = []
        for i in range(N):
            processes.append(
                Process(target=walkGGPGG,
                        args=(
                        walksgene,gene_gene,gene_pathway,pathway_gene, geneset[i], 'p' + str(i))))
        for i in range(N):
            processes[i].start()
        for i in range(N):
            processes[i].join()
        walks = walks + list(walksgene)

        #alks += walkGGG([],gene_gene, walknum,gene_gene.keys(),'p0')
        #walks += walkSVGGVS([],gene_wgs,wgs_sample,sample_wgs,var_gene,gene_gene,gene_gene.keys(),'p0')
        #walks += walkGGPGG([],gene_gene,gene_pathway,pathway_gene,gene_gene.keys(),'p0')
        print('PPI',len(walks))
        #model.train(walks, epochs=15, total_examples=model.corpus_count)
    return walks


def walkDurg(gene_wgs,wgs_sample,sample_wgs,var_gene):
    walk = []
    gdsc = pd.read_csv('./KRAS_subnetwork/GDSC/GDSC.csv',index_col=0)
    durg_gene = dict()
    gene_durg = dict()
    durgable = set()
    variants = set()
    for d in gdsc.index:
        #print(gdsc.loc[d,'targets'])
        genes = set(gdsc.loc[d,'targets'].split('|'))
        genes = {'gene_'+x for x in genes}
        genes = genes & set(gene_wgs.keys())
        durgable = durgable | genes
        if len(genes) != 0:
            durg_gene['drug_'+str(d)]=genes
            for g in genes:
                variants = variants | set(gene_wgs[g])
                if g not in gene_durg:
                    gene_durg[g] = set()
                gene_durg[g].add('drug_'+str(d))
    print('Drug num',len(durg_gene))
    for s in samples:
        for vt in vnum.keys():
            if len(sample_wgs[s][vt] & variants)!=0:
                for v in sample_wgs[s][vt] & variants:
                    for g in var_gene[v] & durgable:
                        for d in gene_durg[g]:
                            for i in range(vnum[vt]):
                                g1 = choice(list(durg_gene[d]))
                                v1 = choice(list(gene_wgs[g1]))
                                s1 = choice(list(wgs_sample[v1]))
                                walk.append([s,v,g,d,g1,v1,s1])
    return walk

def train(sample_wgs, var_gene, gene_pathway, pathways_relation, pathway_gene, walknum, dim):
    global model

    walks=walk(sample_wgs, var_gene, gene_pathway, pathways_relation, pathway_gene, walknum)
    del sample_wgs, var_gene, gene_pathway, pathways_relation, pathway_gene
    '''
    gc.collect()
    f = open(result_path+notes + '/walks.pkl', 'wb')
    pickle.dump(walks, f)
    f.close()
    #global model
    f = open(result_path + notes + '/walks.pkl', 'rb')
    walks = pickle.load(f)'''
    print('walk done')
    model = Word2Vec(walks, size=dim, window=window, min_count=0, sg=1, negative=5, workers=16, iter=15)
    print('train done')
    model.save(result_path + notes + '/trained_model.model')
    model.wv.save_word2vec_format(result_path + notes + '/vecs.txt')
    print('model save at '+result_path + notes)
    del walks
    '''
    for i in range(10):
        print('ITER:',i+1)
        walks = walk(sample_wgs, var_gene, gene_pathway, pathways_relation, pathway_gene, walknum)
        model.train(walks, epochs=15, total_examples=model.corpus_count)
        model.wv.save_word2vec_format(result_path+notes+'/vecs.txt')
    '''

def importvecs(path,dim):
    vecs = dict()
    with open(path+'/vecs.txt', 'r') as f:
        next(f)
        for line in f:
            line = line.split()
            vecs[' '.join(line[:-dim])] = [float(x) for x in line[-dim:]]
    return vecs

def samplecluster(samples, vecs):
    sample_info = pd.read_csv('./patients_info.csv')
    sample_info.index = sample_info['ID']
    types = {'PDAC','ASC','ACC','IPMN','SPT'}
    samplesvecs = {sample:vecs[sample] for sample in samples if sample_info.loc[sample.split('_')[1],'Patho_type'] in types}
    samples = list(samplesvecs.keys())
    print(len(samples))
    vec = pd.DataFrame(samplesvecs)
    vcorr = pd.DataFrame(0,columns=samples,index=samples)
    for i in samples:
        for j in samples:
            vcorr.loc[i,j]=cosine_similarity([vecs[i],vecs[j]])[0][1]
    print(vcorr)
    #vcorr = vec.corr()
    print('calculate corr done')

    vcorr.to_csv(result_path+ notes + '/sample_simi.csv', columns=samples, index=True)

    plt.figure()
    sns.clustermap(vcorr,
                   vmax=0.8,
                   method='ward',
                   #col_colors=[p_color[sample_info.loc[x,'Patho_type']] for x in samples],
                   cmap="hot_r")

    plt.savefig(result_path+notes+'/samplecluster.jpg')

def samplepredict(samples, vecs):

    print(PredictTrait)
    print(sample_test)
    if PredictTrait == 'Tumor_type':
        traitset = ['IA','IB','IIA','IIB','III','IV']
    elif PredictTrait == 'KRAStype':
        traitset = ['G12D','G12V','G12C','G12R','KRASWT','Q61H']
    else:
        traitset = ['G12D', 'G12V', 'G12C', 'G12R', 'KRASWT', 'Q61H']
    traits = pd.read_csv(Traits_path, index_col=1)
    sample_trait_simi = pd.DataFrame(index=samples,columns=traitset)
    for sample in samples:
        for trait in traitset:
            if 'trait_'+trait in vecs:
                sample_trait_simi.loc[sample,trait] = \
                    cosine_similarity([vecs[sample],vecs['trait_'+trait]])[0][1]
    precise = 0
    for sample in sample_test:
        traitset.sort(key=lambda x:  sample_trait_simi.loc[sample,x],reverse=True)
        if traitset[0] == \
                traits.loc[sample[7:], PredictTrait]:
            precise += 1
    print(precise/len(sample_test))
    sample_trait_simi.to_csv(result_path+notes+'/traits_simi.csv')

    for trait in sample_trait_simi.columns:
        sample_trait_simi[trait] = stats.zscore(pd.DataFrame(sample_trait_simi[trait],dtype=float))
    precise = 0
    for sample in sample_test:
        traitset.sort(key=lambda x:  sample_trait_simi.loc[sample,x],reverse=True)
        if traitset[0] == \
                traits.loc[sample[7:],PredictTrait]:
            precise += 1
        print(sample,traitset[0],traits.loc[sample[7:],PredictTrait])
    print(precise / len(sample_test))
    sample_trait_simi.to_csv(result_path + notes + '/traits_simi_zscored.csv')

def testppi(vecs,ppi_test,ppi_neg):
    p = []
    s = []
    with open(ppi_test,'r') as f:
        for l in f:
            g0 = l[:-1].split('\t')[0]
            g1 = l[:-1].split('\t')[1]
            if 'gene_'+g0 not in vecs or 'gene_'+g1 not in vecs:
                continue
            s.append(1)
            p.append(cosine_similarity([vecs['gene_'+g0],vecs['gene_'+g1]])[0][1])
    with open(ppi_neg,'r') as f:
        for l in f:
            g0 = l[:-1].split('\t')[0]
            g1 = l[:-1].split('\t')[1]
            if 'gene_'+g0 not in vecs or 'gene_'+g1 not in vecs:
                continue
            s.append(0)
            p.append(cosine_similarity([vecs['gene_'+g0],vecs['gene_'+g1]])[0][1])
    return p,s

def getargs(argfile):
    wgsdata_path = './data/wgs/'
    pathway_path = './data/pathway/'
    ppi_path = './data/ppi/physicalppi.txt'
    result_path = './data/results/'
    drug_path ='./data/GDSC/GDSC.csv'
    source_path = './data'
    walknum = 5000
    L = 10
    dim = 128
    window = 3

    paths = {'wgsdata_path':wgsdata_path,'pathway_path':pathway_path,
             'ppi_path':ppi_path,'result_path':result_path,
             'drug_path':drug_path,'source_path':source_path}
    args = {'walknum':walknum,'L':L,
             'dim':dim,'window':window,}
    return paths, args

if __name__ == '__main__':
    paths, args = getargs('./args_profile.txt')
    wgsdata_path = paths['wgsdata_path']
    pathway_path = paths['pathway_path']
    ppi_path = paths['ppi_path']
    result_path = paths['result_path']
    drug_path = paths['drug_path']
    source_path = paths['source_path']
    walknum = args['walknum']
    L = args['L']
    dim = args['dim']
    window = args['window']
    notes = 'dim%d_window%d_L%d_dcnv%d_coding%d_sv%d_cnv%d' % (
    dim, window, L, vnum['cnv'], vnum['MutaC'], vnum['sv'],vnum['ncv'])

    if not notes in os.listdir(result_path):
        os.mkdir(result_path+notes)
    sample_var, var_gene = inputwgs(wgsdata_path)

    wgsnodes= getnodes(sample_var, var_gene)
    f = open(sourcepath+'/sample_wgs.pkl', 'wb')
    pickle.dump(sample_var, f)
    f.close()
    f = open(sourcepath+'/var_gene.pkl', 'wb')
    pickle.dump(var_gene, f)
    f.close()
    print('import wgs done')
    f = open(sourcepath+'/nodes.pkl', 'wb')
    gene_wgs, wgs_sample = inverse(sample_var, var_gene)

    gene_pathway, pathways_relation, pathway_gene, pathwaynodes = importpathways(pathwaypath,set(gene_wgs.keys()))
    print('import pathways done')

    pickle.dump({'wgsnodes': wgsnodes, 'pathwaynodes': pathwaynodes}, f)
    f.close()
    samples = list(sample_var.keys())

    train(sample_var, var_gene, gene_pathway, pathways_relation, pathway_gene, walknum, dim)

    print(notes)