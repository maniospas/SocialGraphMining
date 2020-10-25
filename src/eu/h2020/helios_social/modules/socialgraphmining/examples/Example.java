package eu.h2020.helios_social.modules.socialgraphmining.examples;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import eu.h2020.helios_social.core.contextualegonetwork.ContextualEgoNetwork;
import eu.h2020.helios_social.core.contextualegonetwork.Interaction;
import eu.h2020.helios_social.core.contextualegonetwork.Node;
import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.Measure;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.communication.CentralizedFederatedAveragingSimulation;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.communication.UncertainAvailabilitySimulation;
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.DifferenceMiner;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RepeatAndReplyMiner;
import eu.h2020.helios_social.modules.socialgraphmining.measures.Average;
import eu.h2020.helios_social.modules.socialgraphmining.measures.HitRate;

public class Example {
	public static class Device {
		private SocialGraphMiner miner;
		private ContextualEgoNetwork contextualEgoNetwork;
		
		public Device(String name) {
			if(contextualEgoNetwork==null) {
				contextualEgoNetwork = ContextualEgoNetwork.createOrLoad("experiment_data\\", name, null);
				contextualEgoNetwork.setCurrent(contextualEgoNetwork.getOrCreateContext("default"));
			}
			/*SwitchableMiner miner = new SwitchableMiner(contextualEgoNetwork);
			miner.createMiner("repeat", RepeatAndReplyMiner.class);
			miner.createMiner("gnn", GNNMiner.class).setDeniability(0.1, 0.1).setRegularizationAbsorbsion(0);
			this.miner = miner;
			miner.setActiveMiner("gnn");*/
			SocialGraphMiner repeatAndReply = new RepeatAndReplyMiner(contextualEgoNetwork);
			GNNMiner gnnMiner = new GNNMiner(contextualEgoNetwork)
					//.setEmbeddingExchangeProtocol(new CentralizedFederatedAveragingSimulation())
					.setEmbeddingExchangeProtocol(new UncertainAvailabilitySimulation(0))
					.setRegularizationAbsorbsion(1)
					.setMinTrainingRelativeLoss(0.0001)
					//.setLSTMDepth(15)
				    .setTrainingExampleDegradation(0.5)
				    .setTrainingExampleRemovalThreshold(0.01)
				    .setDeniability(0, 0);
			this.miner = new DifferenceMiner(gnnMiner, repeatAndReply, 1);
			//this.miner = repeatAndReply;
		}
		public String getName() {
			return miner.getContextualEgoNetwork().getEgo().getId();
		}
		public void send(Device other) {
			Interaction interaction = miner.getContextualEgoNetwork()
					.getCurrentContext()
					.getOrAddEdge(miner.getContextualEgoNetwork().getEgo(), miner.getContextualEgoNetwork().getOrCreateNode(other.getName(), null))
					.addDetectedInteraction(null);
			other.receive(this, miner.getModelParameters(interaction));
		}
		protected void receive(Device other, String parameters) {
			Interaction interaction = miner.getContextualEgoNetwork()
					.getCurrentContext()
					.getOrAddEdge(miner.getContextualEgoNetwork().getOrCreateNode(other.getName(), null), miner.getContextualEgoNetwork().getEgo())
					.addDetectedInteraction(null);
			miner.newInteraction(interaction, parameters, InteractionType.RECEIVE);
			other.receiveAck(this, miner.getModelParameters(interaction));
		}
		protected void receiveAck(Device other, String parameters) {
			ArrayList<Interaction> interactions = miner.getContextualEgoNetwork()
					.getCurrentContext()
					.getOrAddEdge(miner.getContextualEgoNetwork().getEgo(), miner.getContextualEgoNetwork().getOrCreateNode(other.getName(), null))
					.getInteractions();
			miner.newInteraction(interactions.get(interactions.size()-1), parameters, InteractionType.RECEIVE_REPLY);
		}
		public Node recommendNextInteraction() {
			HashMap<Node, Double> interactionScores = miner.recommendInteractions(miner.getContextualEgoNetwork().getCurrentContext());
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
		Utils.development = false;
		HashMap<String, Device> devices = new HashMap<String, Device>();
		BufferedReader edgeReader = new BufferedReader(new FileReader(new File("datasets/fb-wosn-friends.edges")));
		String line = null;
		ArrayList<String> interactions = new ArrayList<String>();
		ArrayList<Double> timestamps = new ArrayList<Double>();
		while((line=edgeReader.readLine())!=null) {
			if(line.startsWith("%") || line.startsWith("#") || line.isEmpty())
				continue;
			String[] splt = line.split(" ");
			if(splt.length<3)
				continue;
			String time = splt[3];
			if(time.equals("0"))
				continue;
			interactions.add(splt[0]+" "+splt[1]+" "+time);
			timestamps.add(Double.parseDouble(time));
		}
		
		
		
		
		
		Measure measure = new Average(new HitRate(3, 12));
		//Measure measure = new Average(new DiscoveryExactRank(1));
		String result = "";
		int currentInteraction = 0;
		for(int i : eu.h2020.helios_social.modules.socialgraphmining.GNN.operations.Sort.sortedIndexes(timestamps)) {
			String[] splt = interactions.get(i).split(" ");
			String u = splt[0];
			String v = splt[1];
			if(u.equals(v))
				continue;
			if(!devices.containsKey(u))
				devices.put(u, new Device(u));
			if(!devices.containsKey(v))
				devices.put(v, new Device(v));
			double evaluation = measure.evaluateSend(devices.get(u).miner,
					devices.get(u).miner.getContextualEgoNetwork().getCurrentContext(),
					devices.get(u).miner.getContextualEgoNetwork().getOrCreateNode(v, null));
			if(currentInteraction==14000)
				break;
			if(currentInteraction%1000==0)
				System.out.println("#"+currentInteraction+": "+evaluation);
			if(currentInteraction%100==0)
				result += ","+evaluation;
			//if(currentInteraction%10000==0)
			//	System.out.println("["+result.substring(1)+"];\n");
			devices.get(u).send(devices.get(v));
			currentInteraction++;
		}
		edgeReader.close();
		result = "["+result.substring(1)+"];\n";
		//System.out.print(result);
	}
}
