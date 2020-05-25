package eu.h2020.helios_social.module.socialgraphmining.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Serializer;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.module.socialgraphmining.GNN.GNNMiner;

public class Example {
	public static class Device {
		private ContextualEgoNetwork contextualEgoNetwork;
		private SocialGraphMiner miner;
		public Device(String name) {
			contextualEgoNetwork = ContextualEgoNetwork.createOrLoad("experiment_data\\", name, null);
			miner = new GNNMiner(contextualEgoNetwork);
			contextualEgoNetwork.setCurrent(contextualEgoNetwork.getOrCreateContext("default"));
		}
		
		public String getName() {
			return contextualEgoNetwork.getEgo().getId();
		}
		
		public void send(Device other) {
			other.receive(this, miner.getModelParameters(null));
		}
		public void receive(Device other, String parameters) {
			Interaction interaction = contextualEgoNetwork
					.getCurrentContext()
					.getOrAddEdge(contextualEgoNetwork.getOrCreateNode(other.getName(), null), contextualEgoNetwork.getEgo())
					.addDetectedInteraction(null);
			miner.newInteraction(interaction, parameters, InteractionType.RECEIVE);
			other.receiveAck(this, miner.getModelParameters(interaction));
		}
		protected void receiveAck(Device other, String parameters) {
			Interaction interaction = contextualEgoNetwork
					.getCurrentContext()
					.getOrAddEdge(contextualEgoNetwork.getEgo(), contextualEgoNetwork.getOrCreateNode(other.getName(), null))
					.addDetectedInteraction(null);
			System.out.println("other "+other.getName());
			System.out.println("other "+contextualEgoNetwork.getOrCreateNode(other.getName(), null).getId());
			miner.newInteraction(interaction, parameters, InteractionType.RECEIVE_REPLY);
		}
	}
	
	
	public static void main(String[] args) throws Exception {
		HashMap<String, Device> devices = new HashMap<String, Device>();
		
		BufferedReader edgeReader = new BufferedReader(new FileReader(new File("datasets/fb-messages.edges")));
		String line = null;
		while((line=edgeReader.readLine())!=null) {
			String[] splt = line.split("\\,");
			String u = splt[0];
			String v = splt[1];
			if(u.equals(v))
				continue;
			if(!devices.containsKey(u))
				devices.put(u, new Device(u));
			if(!devices.containsKey(v))
				devices.put(v, new Device(v));
			devices.get(u).send(devices.get(v));
		}
		edgeReader.close();
	}
}
