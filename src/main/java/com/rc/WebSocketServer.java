package com.rc;

import java.io.IOException;

import org.eclipse.jetty.io.RuntimeIOException;
import org.eclipse.jetty.websocket.api.Session;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketClose;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketConnect;
import org.eclipse.jetty.websocket.api.annotations.OnWebSocketMessage;
import org.eclipse.jetty.websocket.api.annotations.WebSocket;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@WebSocket
public class WebSocketServer {
	private static Logger log = LoggerFactory.getLogger(WebSocketServer.class);

	private volatile Thread t = null ;
	public static Session sender ;

	@OnWebSocketConnect
	public void onConnect(Session user) throws Exception {
		sender = user ;
		log.info("Client WS connected" ) ;
	}

	@OnWebSocketClose
	public void onClose(Session user, int statusCode, String reason) {
		log.info("Client WS finished" ) ;
		sender = null ;
	}

	@OnWebSocketMessage
	public void onMessage( Session user, String message) {
		if( "hi".equalsIgnoreCase(message) ) {
			begin() ;
		}
		log.info("Client WS msg received" ) ;
	}

	public static void send( String msg ) {
		if( sender != null && sender.isOpen() ) {
			try {
				sender.getRemote().sendString( msg ) ;
			} catch( IOException ioe ) {
				throw new RuntimeIOException( ioe ) ;
			}
		}
	}

	public void begin() {
		if( t != null ) {
			t = new Thread( this::loop ) ;
			t.start();
		}
	}

	public void loop() {
		try {
			while( true ) {    		
				Thread.sleep( 5000 ) ;
				WebSocketServer.send( "hi" ); ;
			}
		} catch( InterruptedException iex ) {
			// ignore
		}
		t = null ;
	}
}

