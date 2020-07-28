package eu.h2020.helios_social.modules.socialgraphmining.tests;

import org.junit.Test;

import eu.h2020.helios_social.core.contextualegonetwork.Utils;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNMiner;
import eu.h2020.helios_social.modules.socialgraphmining.GNN.GNNNodeData;
import org.junit.Assert;

public class GNNMinerTest extends BaseMinerTestFunctionalities {
	
	public GNNMinerTest() {
	}
	
	@Test(expected = Exception.class)
	public void shouldExpectCEN() {
		Utils.development = true;
		new GNNMiner(null);
	}
	
	@Test(expected = Exception.class)
	public void shouldRecommendOnlyForContext() {
        getDevice("A").getMiner().recommendInteractions(null);
	}
	
	@Test
	public void shouldNotRecommendAnythingForEmptyContext() {
		Assert.assertEquals(getDevice("A").recommendInteractionsInCurrentContext().size(), 0);
	}
	
	@Test
	public void shouldRecommendTheOnlySentInteraction() {
		getDevice("A").send(getDevice("B"));
		Assert.assertEquals(argmax(getDevice("A").recommendInteractionsInCurrentContext()), "B");
	}

	@Test
	public void shouldRecommendTheOnlyReceivedInteraction() {
		getDevice("A").send(getDevice("B"));
		Assert.assertEquals(argmax(getDevice("B").recommendInteractionsInCurrentContext()), "A");
	}
	
	@Test
	public void shouldRecommendTheMostRecentInteractionWhenInDoubt() {
		getDevice("A").send(getDevice("B"));
		getDevice("C").send(getDevice("A"));
		Assert.assertEquals(argmax(getDevice("A").recommendInteractionsInCurrentContext()), "C");
	}
	
	@Test
	public void trainingShouldOccurOnEveryInteraction() {
		getDevice("A").send(getDevice("B"));
		String originalEmbeddingsOfBinA = getDevice("A").getMiner().getContextualEgoNetwork()
				.getOrCreateNode("B", null)
				.getOrCreateInstance(GNNNodeData.class)
				.getEmbedding()
				.toString();
		getDevice("C").send(getDevice("A"));
		getDevice("B").send(getDevice("A"));
		String newEmbeddingsOfBinA = getDevice("A").getMiner().getContextualEgoNetwork()
				.getOrCreateNode("B", null)
				.getOrCreateInstance(GNNNodeData.class)
				.getEmbedding()
				.toString();
		Assert.assertTrue(!originalEmbeddingsOfBinA.equals(newEmbeddingsOfBinA));
	}
}
