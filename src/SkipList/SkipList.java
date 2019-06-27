package SkipList;

import java.util.*;
import java.util.concurrent.ConcurrentSkipListMap;

public class SkipList<T> {
	
	public SkipList() {
		// TODO Auto-generated constructor stub
	}
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		List<Node> list = new ArrayList<>();
		list.add(new Node(2, 3));
		list.add(new Node(5, 1));
		list.add(new Node(1, 1));
		//Collections.sort(list);
		SkipList<Node> test = new SkipList();
		ConcurrentSkipListMap<Node, Integer> current = new ConcurrentSkipListMap<Node, Integer>();
		current.put(new Node(2, 3), 12);
		current.put(new Node(4, 1), 12);
		current.put(new Node(1, 3), 12);
		for(Node s:current.keySet()) {
			
		}
		HashMap<Node, Integer> hash = new HashMap<Node, Integer>();
		StringBuilder sb = new StringBuilder();
	}

}

