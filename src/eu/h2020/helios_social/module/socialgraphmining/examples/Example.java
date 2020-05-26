package eu.h2020.helios_social.module.socialgraphmining.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.HashMap;
import java.util.Map.Entry;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.module.socialgraphmining.Measure;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.module.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.module.socialgraphmining.GNN.GNNMiner;
import eu.h2020.helios_social.module.socialgraphmining.measures.Accumulate;
import eu.h2020.helios_social.module.socialgraphmining.measures.HitRate;

public class Example {
	public static class Device {
		private ContextualEgoNetwork contextualEgoNetwork;
		private SocialGraphMiner miner;
		public Device(String name) {
			contextualEgoNetwork = ContextualEgoNetwork.createOrLoad("experiment_data\\", name, null);
			//miner = new RepeatMiner(contextualEgoNetwork); // more informed baseline to evaluate against
			miner = new GNNMiner(contextualEgoNetwork);
			contextualEgoNetwork.setCurrent(contextualEgoNetwork.getOrCreateContext("default"));
		}
		public String getName() {
			return contextualEgoNetwork.getEgo().getId();
		}
		public void send(Device other) {
			other.receive(this, miner.getModelParameters(null));
		}
		protected void receive(Device other, String parameters) {
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
			miner.newInteraction(interaction, parameters, InteractionType.RECEIVE_REPLY);
		}
		public Node recommendNextInteraction() {
			HashMap<Node, Double> interactionScores = miner.recommendInteractions(contextualEgoNetwork.getCurrentContext());
			double bestScore = 0;
			Node bestNode = null;
			for(Entry<Node, Double> interactionScore : interactionScores.entrySet()) 
				if(interactionScore.getValue() > bestScore) {
					bestScore = interactionScore.getValue();
					bestNode = interactionScore.getKey();
				}
			return bestNode;
		}
	}
	
	
	public static void main(String[] args) throws Exception {
		HashMap<String, Device> devices = new HashMap<String, Device>();
		BufferedReader edgeReader = new BufferedReader(new FileReader(new File("datasets/fb-messages.edges")));
		String line = null;
		Measure measure = new Accumulate(new HitRate(3));
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
			double evaluation = measure.evaluateSend(devices.get(u).miner,
					devices.get(u).contextualEgoNetwork.getCurrentContext(),
					devices.get(u).contextualEgoNetwork.getOrCreateNode(v, null));
			System.out.println("EVALUATION: "+evaluation);
			devices.get(u).send(devices.get(v));
		}
		edgeReader.close();
	}
}
