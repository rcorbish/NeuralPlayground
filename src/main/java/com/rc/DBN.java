package com.rc;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.Reader;
import java.io.Writer;
import java.nio.file.Files;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

import org.apache.commons.collections.Bag;
import org.apache.commons.collections.bag.HashBag;
import org.canova.api.io.WritableConverter;
import org.canova.api.io.converters.WritableConverterException;
import org.canova.api.io.data.IntWritable;
import org.canova.api.records.reader.RecordReader;
import org.canova.api.records.reader.impl.CSVRecordReader;
import org.canova.api.split.FileSplit;
import org.canova.api.writable.Writable;
import org.deeplearning4j.datasets.canova.RecordReaderDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.datasets.iterator.impl.ListDataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.conf.layers.RBM;
import org.deeplearning4j.optimize.api.IterationListener;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class DBN extends Model {
	private static Logger log = LoggerFactory.getLogger(DBN.class);

	public DBN(File trainingData, File testData, File configDir ) throws IOException {
		super( trainingData, testData, configDir ) ;
		numInputs = 1000 ;
	}

	List<String> top1000Words ;

	protected List<String> getTop1000Words() {
		return top1000Words ; 
	}

	@SuppressWarnings("unchecked")
	public void train() throws Exception {
		int iterations = 1;
		int listenerFreq = iterations/5;

		Bag dict = new HashBag() ;

		List<String> lines = Files.readAllLines( this.trainingData.toPath() ) ;

		for( String s : lines ) {
			for( String t : s.trim().split("\\s" ) ) {
				if( t.trim().length() > 0 ) {
					dict.add(t) ;
				}
			}
		}
		List<String> topWords = new ArrayList<>( dict.size() ) ;
		topWords.addAll( dict ) ;
		topWords.sort( new Comparator<String>() {
			@Override
			public int compare(String o1, String o2) {
				return dict.getCount(o1) - dict.getCount(o2) ;
			}
		});
		top1000Words = topWords.subList(0, Math.min(numInputs, topWords.size() ) ) ;


		getModel().setListeners(Arrays.asList((IterationListener) new ScoreIterationListener(listenerFreq)));

		log.info("Load data from " + trainingData );

		List<DataSet> al = new ArrayList<>( lines.size() ) ;

		for( String s : lines ) {
			INDArray features = Nd4j.create(numInputs) ;
			for( String t : s.trim().split("\\s" ) ) {
				int n = dict.getCount( t ) ;
				int ix = top1000Words.indexOf( t ) ;
				if( ix>0 && n>0 ) {
					features.putScalar(ix, 1 ) ;
				}
			}
			DataSet ds = new DataSet( features, features ) ;
			al.add(ds) ;
		}
		DataSetIterator iter = new ListDataSetIterator( al, 250 ) ;

		log.info("Train model....");
		while(iter.hasNext()) {
			DataSet next = iter.next();
			getModel().fit(new DataSet(next.getFeatureMatrix(),next.getFeatureMatrix()));
		}
	}


	public Evaluation<String> test() throws Exception {

		List<String> lines = Files.readAllLines( this.trainingData.toPath() ) ;
		log.info("Load data from " + trainingData );

		List<DataSet> al = new ArrayList<>( lines.size() ) ;

		for( String s : lines ) {
			INDArray features = Nd4j.create(numInputs) ;
			for( String t : s.trim().split("\\s" ) ) {
				int ix = top1000Words.indexOf( t ) ;
				if( ix>0 ) {
					features.putScalar(ix, 1 ) ;
				}
			}
			DataSet ds = new DataSet( features, features ) ;
			al.add(ds) ;
		}
		DataSetIterator iter = new ListDataSetIterator( al, 250 ) ;

		Evaluation<String> eval = new Evaluation<>( numInputs );

		int n = 0 ;
		try {
			while(iter.hasNext()) {
				n++ ;
				DataSet ds = iter.next();
				INDArray predict2 = getModel().output(ds.getFeatureMatrix(), false);
				eval.eval(predict2, predict2);
			}
		} catch( Throwable t ) {
			t.printStackTrace();
			log.info( "Managed to load " + n + " items" );
		}

		log.info(eval.stats());
		log.info("All Done");
		return eval ;
	}

	@Override
	public MultiLayerConfiguration createModelConfig() throws IOException {

		MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
				.seed(100)
				.iterations(300)
				.optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
				.list(10)
				.layer(0, new RBM.Builder().nIn(numInputs).nOut(800).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(1, new RBM.Builder().nIn(800).nOut(600).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(2, new RBM.Builder().nIn(600).nOut(300).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(3, new RBM.Builder().nIn(300).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(4, new RBM.Builder().nIn(100).nOut(50).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //encoding stops
				.layer(5, new RBM.Builder().nIn(50).nOut(100).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build()) //decoding starts
				.layer(6, new RBM.Builder().nIn(100).nOut(300).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(7, new RBM.Builder().nIn(300).nOut(600).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(8, new RBM.Builder().nIn(600).nOut(800).lossFunction(LossFunctions.LossFunction.RMSE_XENT).build())
				.layer(9, new OutputLayer.Builder(LossFunctions.LossFunction.RMSE_XENT).nIn(800).nOut(numInputs).build())
				.pretrain(true).backprop(true)
				.build();
		return conf ;
	}

	public void loadModel( boolean loadUpdater ) throws IOException, ClassNotFoundException {
		super.loadModel(loadUpdater);
		
		if( configDir != null ) {
			File f = new File( configDir, "top1000.txt" ) ;
			try( BufferedReader br = new BufferedReader(new FileReader( f ) ) ) {
				List<String> l = getTop1000Words() ;
				l.clear(); 
				for( String s=br.readLine() ; s!=null ; s=br.readLine() ) {
					l.add(s) ;
				}
			}
		}
	}

	public void saveModel( boolean saveUpdater ) throws IOException {
		super.saveModel( saveUpdater ) ;

		if( configDir != null ) {
			File f = new File( configDir, "top1000.txt" ) ;
			try( Writer w = new FileWriter( f ) ) {
				for( String s : getTop1000Words() ) {
					w.append( s ).append( System.getProperty("line.separator") ) ;
				}
				w.flush();
			}

		}
	}
}


