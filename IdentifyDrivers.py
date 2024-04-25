import pandas as pd
import pickle
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import binom
import Embedding
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
import adjustText

vnum = {'cnv':1,'sv':1,'MutaC':4,'ncv':2,'peak':2}
types = ['cnv','sv','MutaC','ncv']
N = 8

def calsimi_sg():
    samples = list(wgsnodes['samples'])
    genes = list(wgsnodes['genes'])
    sgsimi = pd.DataFrame(cosine_similarity(model.wv[genes], model.wv[samples]), index=genes, columns=samples)
    sgsimi.to_csv(result_path + notes + '/gene_sample_simi.csv')
    sssimi = pd.DataFrame(cosine_similarity(model.wv[samples], model.wv[samples]), index=samples, columns=samples)
    sssimi.to_csv(result_path + notes + '/sample_sample_simi.csv')
    
    codingNUM = pd.read_csv(source_path + '/gene_sample_num_coding.csv',index_col=0)
    genes = list(codingNUM.index)
    codingNUM.index = ['gene_'+g for g in codingNUM.index]
    codingNUM.columns = ['sample_'+s for s in codingNUM.columns]
    codingsimi = codingNUM * (sgsimi.loc[codingNUM.index,codingNUM.columns]+1)
    codingsimi.to_csv(result_path + notes + '/gene_sample_coding_simi.csv')
    coding_genes = pd.DataFrame({'Gene':genes,
                                 'Mutation_Num':codingNUM.sum(axis=1),
                                 'Raw_score':codingsimi.sum(axis=1)/len(samples)})
    coding_genes = coding_genes.sort_values('Raw_score',ascending=False)
    coding_genes.to_csv(result_path + notes + '/gene_coding_score.csv')
    print(coding_genes.head(10))

    noncodingNUM = pd.read_csv(source_path + '/gene_sample_num_noncoding.csv', index_col=0)
    genes = list(noncodingNUM.index)
    noncodingNUM.index = ['gene_' + g for g in noncodingNUM.index]
    noncodingNUM.columns = ['sample_' + s for s in noncodingNUM.columns]
    noncodingsimi = noncodingNUM * (sgsimi.loc[noncodingNUM.index, noncodingNUM.columns] + 1)
    noncodingsimi.to_csv(result_path + notes + '/gene_sample_noncoding_simi.csv')
    noncoding_genes = pd.DataFrame({'Gene': genes,
                                 'Mutation_Num': noncodingNUM.sum(axis=1),
                                 'Raw_score': noncodingsimi.sum(axis=1) / len(samples)})
    noncoding_genes = noncoding_genes.sort_values('Raw_score', ascending=False)
    noncoding_genes.to_csv(result_path + notes + '/gene_noncoding_score.csv')
    print(noncoding_genes.head(10))

def constructnet():
    coding_genes = pd.read_csv(result_path + notes + '/gene_coding_score.csv', index_col=0)
    genes = list(coding_genes['Gene'].values)

    networkppi = pd.DataFrame(0, index=genes, columns=genes)
    with open(ppi_path, 'r') as f:
        for line in f:
            l = line[:-1].split('\t')
            if l[1] in genes and l[0] in genes:
                networkppi.loc[l[0], l[1]] = 1
                networkppi.loc[l[1], l[0]] = 1
    networkppi.to_csv(source_path+'/ppi.csv')

    networkpathway = pd.DataFrame(0, index=genes, columns=genes)
    with open(pathway_path+'/ReactomePathways.gmt','r') as f:
        for line in f:
            l = line[:-1].split('\t')
            gset = set(l[2:]) & set(genes)
            if len(gset)>1:
                w = 2 / (len(gset) * (len(gset) - 1))
                for i in gset:
                    for j in gset:
                        if i != j:
                            networkpathway.loc[i,j] += w
    networkpathway.to_csv(source_path + '/pathwaynet.csv')


def netprop():
    coding_genes = pd.read_csv(result_path + notes + '/gene_coding_score.csv', index_col=0)
    genes = list(coding_genes['Gene'].values)
    ggsimi = pd.DataFrame(cosine_similarity(model.wv[['gene_'+g for g in genes]],
                                            model.wv[['gene_'+g for g in genes]]),
                          index=genes, columns=genes)
    print(ggsimi.shape)
    networkppi = pd.read_csv(source_path+'/ppi.csv',index_col=0)
    networkpathway = pd.read_csv(source_path + '/pathwaynet.csv', index_col=0)
    network_prop = (ggsimi+1) * (networkppi.loc[genes,genes] + networkpathway.loc[genes,genes])
    network_prop = network_prop.loc[(network_prop != 0).any(axis=0), (network_prop != 0).any(axis=0)]
    '''
    for gene in network_prop.index:
        if all(network_prop[gene]==0):
            network_prop = network_prop.drop(gene)
            network_prop = network_prop.drop(gene,axis=1)'''
    print(network_prop.shape)
    seedset = {'KRAS', 'TP53', 'SMAD4', 'CDKN2A', 'ARID1A', 'RNF43', 'GNAS', 'KMT2C', 'TGFBR2', 'RBM10', 'LRP1B',
               'ERBB4', 'KDM6A', 'ACVR2A', 'TBX3', 'EXT2', 'NF1', 'ARID1B', 'GNAQ'}
    driverlist = {'ACVR1B', 'ACVR2A', 'ARID1A', 'ARID1B', 'ATM', 'BCORL1', 'BRAF', 'BRCA1', 'BRCA2', 'CALD1',
                  'CDK6', 'CDKN2A', 'DISP2', 'ERBB2', 'FBLN2', 'FBXW7', 'FGFR2', 'GATA6', 'GNAS', 'HIVEP1', 'ITPR3',
                  'JAG1', 'KALRN', 'KAT8', 'KDM6A', 'KRAS', 'MACF1', 'MAP2K4', 'MARK2', 'MET', 'MLH1', 'KMT2B', 'KMT2C',
                  'MSH2', 'MYC', 'MYCBP2', 'NBEA', 'NF2', 'PLXNA1', 'PALB2', 'PBRM1', 'PIK3CA', 'PIK3R3', 'PLXNB2',
                  'PREX2',
                  'RBM10', 'RIPK4', 'RNF43', 'ROBO1', 'ROBO2', 'RPA1', 'RREB1', 'SDK2', 'SETD2', 'SF3B1', 'SIN3B',
                  'SLIT2', 'SMAD3', 'SMAD4', 'SMARCA2', 'SMARCA4', 'SOX9', 'SPTB', 'STK11', 'TGFBR1', 'TGFBR2', 'TLE4',
                  'TP53', 'TP53BP2', 'U2AF1', 'ZFP36L2', 'CTNNB1', 'DUSP6', 'FGFR4', 'JAK1', 'NF1', 'PRSS1', 'RREB1',
                  'SPRED1'}
    genelist = (driverlist | seedset) & set(network_prop.index)
    cancergene = pd.read_csv('../CancerGeneCosmic.csv', index_col=0)
    Cancergene = set(cancergene.index)
    LNC = {'TTN', 'MUC16', 'OBSCN', 'AHNAK2', 'SYNE1', 'FLG', 'MUC5B', 'DNAH17', 'PLEC', 'DST', 'SYNE2', 'NEB', 'HSPG2',
           'LAMA5', 'AHNAK', 'HMCN1', 'USH2A', 'DNAH11', 'MACF1', 'MUC17', 'DNAH5', 'GPR98', 'FAT1', 'PKD1', 'MDN1',
           'RNF213', 'RYR1', 'DNAH2', 'DNAH3', 'DNAH8', 'DNAH1', 'DNAH9', 'ABCA13', 'APOB', 'SRRM2', 'CUBN', 'SPTBN5',
           'PKHD1', 'LRP2', 'FBN3', 'CDH23', 'DNAH10', 'FAT4', 'RYR3', 'PKHD1L1', 'FAT2', 'CSMD1', 'PCNT', 'COL6A3',
           'FRAS1', 'FCGBP', 'DNAH7', 'RP1L1', 'PCLO', 'ZFHX3', 'COL7A1', 'LRP1B', 'FAT3', 'EPPK1', 'VPS13C', 'HRNR',
           'MKI67', 'MYO15A', 'STAB1', 'ZAN', 'UBR4', 'VPS13B', 'LAMA1', 'XIRP2', 'BSN', 'KMT2C', 'ALMS1', 'CELSR1',
           'TG', 'LAMA3', 'DYNC2H1', 'KMT2D', 'BRCA2', 'CMYA5', 'SACS', 'STAB2', 'AKAP13', 'UTRN', 'VWF', 'VPS13D',
           'ANK3', 'FREM2', 'PKD1L1', 'LAMA2', 'ABCA7', 'LRP1', 'ASPM', 'MYOM2', 'PDE4DIP', 'TACC2', 'MUC2', 'TEP1',
           'HELZ2', 'HERC2', 'ABCA4'}
    LNC = LNC - genelist - Cancergene

    print(len(genelist))
    seedvec = list()
    stand = list()

    print(len(genelist))
    print(genelist)
    for gene in network_prop.index:
        seedvec.append(coding_genes.loc['gene_'+gene, 'Raw_score'])
        if gene in genelist:
            stand.append(1)
        else:
            stand.append(0)
        network_prop.loc[gene] = network_prop.loc[gene] / sum(network_prop.loc[gene])
    seedvec = np.array(seedvec)

    vec = seedvec

    ad = np.array(network_prop)

    fpr, tpr, threshold = roc_curve(stand, list(vec))
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    temp = roc_auc
    pbest = 0
    pmax = 0
    prcbest = 0
    vecbest = seedvec

    for p in [x / 100 for x in range(101)]:
        step = 0
        vec = (1 - p) * np.dot(vec, ad) + p * np.array(seedvec)
        fpr, tpr, threshold = roc_curve(stand, list(vec))
        auroc = auc(fpr, tpr)

        while abs(auroc - temp) > 10 ** (-50) and step < 5000:
            temp = auroc
            vec = (1 - p) * np.dot(vec, ad) + p * np.array(seedvec)
            fpr, tpr, threshold = roc_curve(stand, list(vec))
            auroc = auc(fpr, tpr)
            step += 1
        pre, rec, threshold = precision_recall_curve(stand, list(vec))
        auprc = auc(rec, pre)
        if temp > pmax:
            pbest = p
            pmax = auroc
            prcbest = auprc
        if p == 0.5:
            vecbest = vec
        print(p, auroc, auprc)
    print(pbest, pmax, prcbest)

    genes = list(network_prop.index)
    out = dict(zip(genes, vecbest))
    for g in out.keys():
        coding_genes.loc['gene_'+g,'Score_afterNP'] = out[g]
        if g in genelist - seedset:
            coding_genes.loc['gene_'+g,'label'] = '**'
        elif g in seedset:
            coding_genes.loc['gene_'+g,'label'] = '*'
        elif g in Cancergene:
            coding_genes.loc['gene_'+g,'label'] = '+'
        elif g in LNC:
            coding_genes.loc['gene_'+g,'label'] = '-'
        else:
            coding_genes.loc['gene_'+g,'label'] = 'N'
    coding_genes.to_csv(result_path + notes + '/gene_coding_score.csv')

def outcodingdriver():
    scores = pd.read_csv(result_path + notes + '/gene_coding_score.csv', index_col=0)
    scores = scores.dropna()
    sample_gene = pd.read_csv(result_path + notes + '/gene_sample_simi.csv', index_col=0)
    sample_gene = sample_gene + 1
    sample_gene = sample_gene.loc[scores.index]
    #simithres = np.percentile(sample_gene.values.reshape(1, -1), 99)
    MutaCNum = scores['Mutation_Num'].values.sum()
    MutaCNum = binom.ppf(0.95, MutaCNum, 1 / len(scores.index)) + 1
    #simithres = simithres * MutaCNum / len(sample_gene.columns)
    simithres = np.percentile(list(scores['Raw_score'].values),97)
    print(MutaCNum)
    print(simithres)
    scores = scores.dropna()
    #scores = scores[scores['label'] != '-']
    CancerGene = [g.strip('\n') for g in open('../cancer_related_gene.txt','r')]
    scores = scores[scores['Gene'].isin(CancerGene)]
    CDs = [g for g in set(scores.index)
           if scores.loc[g, 'Mutation_Num'] >= MutaCNum and
           scores.loc[g, 'Raw_score'] >= simithres and
           scores.loc[g, 'Score_afterNP'] >= simithres
           ]

    CDs.sort(key=lambda x: scores.loc[x, 'Raw_score'], reverse=True)
    CMutas = scores.loc[CDs]
    print(CMutas)
    print(len(CMutas.loc[CMutas['label'] == '*'].index),
          len(CMutas.loc[CMutas['label'] == '**'].index),
          len(CMutas.loc[CMutas['label'] == '+'].index), )
    CMutas.to_csv(result_path + notes + '/CodingDrivers.csv', index=True)
    driverlist = {'ACVR1B', 'ACVR2A', 'ARID1A', 'ARID1B', 'ATM', 'BCORL1', 'BRAF', 'BRCA1', 'BRCA2', 'CALD1',
                  'CDK6', 'CDKN2A', 'DISP2', 'ERBB2', 'FBLN2', 'FBXW7', 'FGFR2', 'GATA6', 'GNAS', 'HIVEP1', 'ITPR3',
                  'JAG1', 'KALRN', 'KAT8', 'KDM6A', 'KRAS', 'MACF1', 'MAP2K4', 'MARK2', 'MET', 'MLH1', 'KMT2B', 'KMT2C',
                  'MSH2', 'MYC', 'MYCBP2', 'NBEA', 'NF2', 'PLXNA1', 'PALB2', 'PBRM1', 'PIK3CA', 'PIK3R3', 'PLXNB2',
                  'PREX2',
                  'RBM10', 'RIPK4', 'RNF43', 'ROBO1', 'ROBO2', 'RPA1', 'RREB1', 'SDK2', 'SETD2', 'SF3B1', 'SIN3B',
                  'SLIT2', 'SMAD3', 'SMAD4', 'SMARCA2', 'SMARCA4', 'SOX9', 'SPTB', 'STK11', 'TGFBR1', 'TGFBR2', 'TLE4',
                  'TP53', 'TP53BP2', 'U2AF1', 'ZFP36L2', 'CTNNB1', 'DUSP6', 'FGFR4', 'JAK1', 'NF1', 'PRSS1', 'RREB1',
                  'SPRED1'}
    plt.figure(figsize=(6, 6))
    plt.scatter(scores['Raw_score'],scores['Score_afterNP'],s=scores['Mutation_Num'],c='gray')
    plt.scatter(CMutas['Raw_score'],CMutas['Score_afterNP'],s=CMutas['Mutation_Num'],c='red',label='Drivers')
    plt.scatter(scores[scores['label']=='*']['Raw_score'],scores[scores['label']=='*']['Score_afterNP'],
                s=scores[scores['label']=='*']['Mutation_Num'],marker='o',lw=0.3,c='none',edgecolor='blue',
                label='Drivers defined by other methods')
    plt.scatter(scores[scores['Gene'].isin(driverlist)]['Raw_score'], scores[scores['Gene'].isin(driverlist)]['Score_afterNP'],
                s=scores[scores['Gene'].isin(driverlist)]['Mutation_Num'], marker='x', lw=0.3, c='blue',
                label='Drivers in other researches')
    text = []
    for g in CMutas.index:
        text.append(plt.text(CMutas.loc[g,'Raw_score'],CMutas.loc[g,'Score_afterNP'],CMutas.loc[g,'Gene']))
    plt.yscale('log')
    plt.xscale('log')
    plt.legend()
    adjustText.adjust_text(text, arrowprops=dict(arrowstyle='-', lw=0.5, color='grey'))
    plt.xlabel('MutScore$^C$(G)')
    plt.ylabel('MutScore$^{C^{NP}}$(G)')
    plt.tight_layout()
    plt.savefig(result_path + notes + '/CodingDrivers_Num.pdf')
    plt.close()

def peakcentric():
    peakinfo = pd.read_csv(wgsdata_path+'/ncvprocess/peak_noncoding_info.csv')
    peakanno = pd.read_csv(wgsdata_path + '/peak2gene.csv')
    peakinfo['MutScore'] = 0
    for i in peakinfo.index:
        for ncv in peak_var['peak_'+peakinfo.loc[i,'Peak']]:
            for s in var_sample[ncv]:
                peakinfo.loc[i,'MutScore'] += 1 + model.wv.similarity(s,'peak_'+peakinfo.loc[i,'Peak'])
        if not 'distal' in peakanno[peakanno['peak']==peakinfo.loc[i,'Peak']]['anno'].values:
            peakinfo.loc[i,'anno'] = 'promoter'
        elif not 'promoter' in peakanno[peakanno['peak'] == peakinfo.loc[i, 'Peak']]['anno'].values:
            peakinfo.loc[i, 'anno'] = 'distal'
        else:
            peakinfo.loc[i, 'anno'] = 'both'

    peakinfo.sort_values('MutScore',ascending=False)
    peakinfo['MutScore'] = peakinfo['MutScore']/len(wgsnodes['samples'])
    peakinfo['chr'] = [p.split('_')[0] for p in peakinfo['Peak']]
    peakinfo['peak_center'] = [int(p.split('_')[1]) + 250 for p in peakinfo['Peak']]
    peakinfo.to_csv(result_path+notes+'/Peaks_info.csv',index=False)

    score_thres = np.percentile(peakinfo['MutScore'], 95)
    num_thres = binom.ppf(0.90, peakinfo['Sample Num'].values.sum(), 1 / len(peakinfo.index))

    driverpeak = peakinfo[(peakinfo['MutScore']>score_thres) & (peakinfo['Sample Num']>num_thres)]
    driverpeak.to_csv(result_path+notes+'/Peaks_driver.csv',index=False)
    drivergene = pd.read_csv(result_path+notes+'/NoncodingDrivers.csv',index_col=0)
    driverpeaknodeset = {'peak_'+p for p in driverpeak['Peak']}
    for g in drivergene.index:
        peaks = gene_edges[g] & driverpeaknodeset
        if len(peaks)!= 0:
            drivergene.loc[g,'with driver peak'] = '|'.join([p[5:] for p in peaks])
            peaks = [p[5:] for p in peaks]
            annos = peakanno[(peakanno['peak'].isin(peaks)) & (peakanno['target gene']==g[5:])]['anno'].values
            if not 'distal' in annos:
                drivergene.loc[g, 'matched peak anno'] = 'promoter'
            elif not 'promoter' in annos:
                drivergene.loc[g, 'matched peak anno'] = 'distal'
            else:
                drivergene.loc[g, 'matched peak anno'] = 'both'
        else:
            drivergene.loc[g,'matched peak anno'] = 'none'
    drivergene.to_csv(result_path + notes + '/NoncodingDrivers.csv')

if __name__=='__main__':
    paths, args = Embedding.getargs('./args_profile.txt')
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
    notes = 'dim%d_window%d_L%d_dcnv%d_coding%d_sv%d_cnv%d' % (dim, window, L, vnum['cnv'], vnum['MutaC'], vnum['sv'], vnum['ncv'])
    model = Word2Vec.load(result_path + notes + '/trained_model.model')
    f = open(source_path + '/wgsdata.pkl', 'rb')
    wgsdata = pickle.load(f)
    f = open(source_path + '/wgsnodes.pkl', 'rb')
    wgsnodes = pickle.load(f)
    sample_wgs, var_edges, var_sample, peak_gene, peak_var, gene_edges = wgsdata
    calsimi_sg()
    constructnet()
    netprop()
    outcodingdriver()
    peakcentric()

