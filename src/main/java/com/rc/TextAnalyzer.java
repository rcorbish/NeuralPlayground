package com.rc;

import java.nio.file.Path;
import java.util.Collection;
import java.util.concurrent.BlockingQueue;

import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class TextAnalyzer extends Model {

	private static Logger log = LoggerFactory.getLogger(TextAnalyzer.class);

	@Override
	public String test( Path testData ) throws Exception {
		return "" ;
	}
	
	@Override
	public void createModelConfig(int numLayers, int numInputs, int numOutputs) {
	}

	@Override
	public BlockingQueue<String> train( Path trainingData ) throws Exception {
		
		log.info("Load & Vectorize Sentences....");
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(trainingData.toFile());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        log.info("Building model....");
        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(2)
                .iterations(10)
                .layerSize(100)
                .seed(42)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        log.info("Fitting Word2Vec model....");
        vec.fit();

        log.info("Writing word vectors to text file....");

        // Write word vectors
        WordVectorSerializer.writeWordVectors(vec, "pathToWriteto.txt");

        log.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("professional", 3);
        System.out.println(lst);
        
        return null ;
	}

}
