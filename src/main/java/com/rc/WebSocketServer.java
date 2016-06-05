package com.rc;

import java.io.IOException;

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
    	log.info("Client WS msg received" ) ;
    }
    
    public static void send( String msg ) throws IOException {
    	if( sender != null && sender.isOpen() ) {
    		sender.getRemote().sendString( msg ) ;
    	}
    }
}

