package com.rc;

import static spark.Spark.get;

import java.util.HashMap;
import java.util.Map;

import org.deeplearning4j.nn.api.Layer;

import spark.ModelAndView;
import spark.Request;
import spark.Response;
import spark.template.jade.JadeTemplateEngine;

public class WebServer {

	Model nn ;
	public WebServer( Model nn ) {
		this.nn = nn ;
		get( "/", this::home, new JadeTemplateEngine() ) ;
	}
	
	public ModelAndView home( Request request, Response response ) {
		Map<String,Object> map = new HashMap<>() ;
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
		
		
		map.put( "labels", nn.labelIndices ) ;
		return new ModelAndView( map, "index" )  ;
	}
}
