import os, json
import pandas as pd
from tqdm import tqdm
import tensorflow as tf

def GenerateOutDir(args):
    outDir = args.outDir
    if outDir is None:
        if os.path.isfile(args.input):
            outDir = args.input.split('/')[0:-1]
        else:
            outDir = args.input
    if not os.path.exists(outDir):
        os.makedirs(outDir)
    return outDir

def IdentifyWithArgs(args):
    # Generate input and output file
    inOutFiles = []
    outDir = GenerateOutDir(args)
    if os.path.isfile(args.input):
        inOutFiles.append(args.input, outDir+'/'+args.input.split('/')[-1].replace('.tsv', '.df.out'))
    else:
        for fileName in os.listdir(args.input):
            inFile = args.input+'/'+fileName
            outFile = outDir+'/'+fileName.replace('.tsv', '.df.out')
            inOutFiles.append((inFile, outFile))

    # Run DeepFavored and save the results
    with open(args.modelDir+'/model.json') as f:
        confArgs = json.load(f)
    features = confArgs['trainDataParamDict']['componentStats']
    with tf.Graph().as_default() as g:#use tf.Graph().as_default() to separate differnt Scope
        with tf.compat.v1.Session() as sess:
            saver = tf.compat.v1.train.import_meta_graph(args.modelDir+'/model.meta')#, clear_devices=True)
            saver.restore(sess, args.modelDir+'/model')
            graph = tf.compat.v1.get_default_graph()
            
            X_tensor = graph.get_tensor_by_name("X:0")
            H_tensor = graph.get_tensor_by_name("Y1_pred:0")
            O_tensor = graph.get_tensor_by_name("Y2_pred:0")
            
            for inFile, outFile in tqdm(inOutFiles):
                df = pd.read_csv(inFile, sep='\t')
                df.dropna(inplace=True)
                df_features = df[features]
                X = df_features.values
                coors = df.Pos.values
                H, O = sess.run([H_tensor, O_tensor], feed_dict={X_tensor: X})
                table = []
                for coor, h, o in list(zip(coors, H, O)):
                    h, o = h[0], o[0]
                    ho = h*o
                    coor, h, o, ho = int(coor), round(h, 6), round(o, 6), round(ho, 6)
                    table.append((coor, h, o, ho))

                table = sorted(table, key=lambda x:x[0])

                with open(outFile, 'w') as f:
                    f.write('Pos\tDF_H\tDF_O\tDF\n')
                    for row in table:
                        row = [str(i) for i in row]
                        f.write('\t'.join(row)+'\n')




