package com.rc;

import static spark.Spark.get;
import static spark.Spark.post;
import static spark.Spark.staticFiles;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;

import com.google.common.io.Files;

import spark.ModelAndView;
import spark.Request;
import spark.Response;
import spark.template.jade.JadeTemplateEngine;

public class WebServer {

	private Model nn ;
	
	public WebServer( Model nn ) {
		this.nn = nn ;
		
    	staticFiles.location(".");
    	staticFiles.expireTime(600);
    	
		get( "/", this::home, new JadeTemplateEngine() ) ;
		get( "/config", this::config ) ;
		post( "/upload-train", this::uploadTrain ) ;
		post( "/upload-test", this::uploadTest ) ;
		post( "/upload-config", this::uploadConfig ) ;
	}

	
	public Object uploadTrain( Request request, Response response ) throws Exception {
		String body = request.body() ;
		nn.test() ;
		return "" ;
	}
	public Object uploadTest( Request request, Response response ) throws Exception {
		File f = File.createTempFile( "test", ".dat" ) ;
		Files.write( request.bodyAsBytes(), f ) ;
		nn.setTestDataFile( f ) ;
		return nn.test().stats() ;
	}
	
	public Object uploadConfig( Request request, Response response ) {
		return "Boo Ya" ;
	}

	public Object config( Request request, Response response ) {
		response.header( "Content-Type", "application/json" );
		return nn.getModel().getLayerWiseConfigurations().toJson() ;
	}
	
	public ModelAndView home( Request request, Response response ) {
		Map<String,Object> map = new HashMap<>() ;
/*
		Layer[] layers = nn.getModel().getLayers() ;
		for( Layer layer : layers ) {
			layer.numParams() ;
		}
		int layerInputs[] = new int[ layers.length + 1 ] ;
		for( int i=0 ; i<layers.length ; i++ ) {
			layerInputs[i] = layers[i].input().columns() ;
		}
		layerInputs[layers.length] = nn.numOutputs ;
		
		map.put( "layerInputs", layerInputs ) ;
		*/
		return new ModelAndView( map, "index" )  ;
	}
}
