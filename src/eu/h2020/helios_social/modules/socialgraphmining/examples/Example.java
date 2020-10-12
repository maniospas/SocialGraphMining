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
import eu.h2020.helios_social.modules.socialgraphmining.SocialGraphMiner.InteractionType;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.DifferenceMiner;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.ProbabilityMiner;
import eu.h2020.helios_social.modules.socialgraphmining.heuristics.RepeatAndReplyMiner;
import eu.h2020.helios_social.modules.socialgraphmining.measures.Accumulate;
import eu.h2020.helios_social.modules.socialgraphmining.measures.Average;
import eu.h2020.helios_social.modules.socialgraphmining.measures.DiscoveryExactRank;
import eu.h2020.helios_social.modules.socialgraphmining.measures.DiscoveryRank;
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
			GNNMiner gnnMiner = new GNNMiner(contextualEgoNetwork).setRegularizationAbsorbsion(0).setLSTMDepth(0)
				     	.setTrainingExampleDegradation(0.5).setTrainingExampleRemovalThreshold(0.01).setDeniability(0, 0)
				     	//.setRegularizationWeight(.1)
				     	.setTrainingExamplePropagation(false);
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
		BufferedReader edgeReader = new BufferedReader(new FileReader(new File("datasets/ia-enron-email-dynamic.edges")));
		String line = null;
		Measure measure = new Average(new HitRate(3, 12));
		//Measure measure = new Average(new DiscoveryExactRank(1));
		String result = "";
		int currentInteraction = 0;
		while((line=edgeReader.readLine())!=null) {
			if(line.startsWith("%") || line.startsWith("#") || line.isEmpty())
				continue;
			String[] splt = line.split(" ");
			if(splt.length<3)
				continue;
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
