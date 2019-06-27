package SkipList;

import java.util.AbstractMap;
import java.util.Comparator;
import java.util.Random;
import java.util.Set;

public class SkipListMap<K,V> extends AbstractMap<K, V> implements Cloneable,java.io.Serializable {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = -8686810942344122479L;
	/**
	 * Node:节点类，用来保存key和value
	 * 
	 * */
	
	/**
	 * 在randomLevel中用到的随机数生成器
	 * */
	private static final Random seedGenerator = new Random();
	
	/**
	 * 用来识别header
	 * */
	private static final Object BASE_HEADER = new Object();
	
	/**
	 * SkipList最顶层的头索引
	 * */
	private HeadInde<K, V> head;
	
	/**
	 * 用于确定map顺序的比较器
	 * */
	private Comparator<? super K> comparator;
	
	private int randomSeed;
	
	
	final void initialize() {
		
		randomSeed = seedGenerator.nextInt() | 0x0100;//ensure nonzero
	}
	
	static class Node<K,V> {
		
		final K key;
		Object value;
		Node<K,V> next;
		Node(K key,Object value,Node<K,V> next){
			this.key = key;
			this.value = value;
			this.next = next;
		}
		
		Node(Node<K,V> next){
			this.key = null;
			this.value = this;
			this.next = next;
		}
		
		boolean isMarker() {
			return value == this;
		}
		
		boolean isBaseHeader() {
			return false;
		}
		
		V getValidValue() {
			Object v = value;
			if(v==this || v==BASE_HEADER)
				return null;
			return (V)v;
		}
		
	}
	
	static class Index<K,V> {
		final Node<K,V> node;
		final Index<K, V> down;
		Index<K,V> right;
		
		Index(Node<K,V> node,Index<K, V> down,Index<K, V> right){
			this.node = node;
			this.down = down;
			this.right = right;
		}
	}
	
	static final class HeadInde<K,V> extends Index<K, V>{
		final int level;
		public HeadInde(Node<K,V> node,Index<K, V> down,Index<K, V> right,int level) {
			super(node, down, right);
			this.level = level;
		}
	}
	
	/*-------------------------- Comparison utilities -----------------------*/
	static final class ComparableUsingComparator<K> implements Comparable<K> {

		@Override
		public int compareTo(K o) {
			// TODO Auto-generated method stub
			return 0;
		}
		
	}
	
	private V doPut(K kkey,V value) {
		return value;
	}
	
	private V doGet(Object key) {
		// TODO Auto-generated method stub
		return null;
	}
	
	final V doRemove(Object okey,Object value) {
		return null;
	}
	
	private int randomLevel() {
		int x = randomSeed;
		x ^= x << 13;
		x ^= x >> 17;
		randomSeed = x ^= x << 5;
		if((x & 80000001)!=0)
			return 0;
		int level = 1;
		while(((x>>>=1)&1)!=0) level++;
		return level;
	}
	
	
	/*-------- Map API methods -------------------*/
	
	@Override
	public boolean containsKey(Object key) {
		// TODO Auto-generated method stub
		return super.containsKey(key);
	}
	
	
	
	
	/**
	 * 
	 * 
	 * */
	@Override
	public boolean containsValue(Object value) {
		// TODO Auto-generated method stub
		return super.containsValue(value);
	}
	
	@Override
	public V get(Object key) {
		return doGet(key);
	}
	
	@Override
	public V put(K key,V value) {
		return value;
	}
	
	@Override
	public int size() {
		return 0;
	}
	
	@Override
	public boolean isEmpty() {
		// TODO Auto-generated method stub
		return super.isEmpty();
	}
	
	@Override
	public void clear() {
		
	}

	/*------- View methods ------*/
	@Override
	public Set<Entry<K, V>> entrySet() {
		// TODO Auto-generated method stub
		return null;
	}
	
	
	/**/
	public boolean equals(Object o) {
		return false;
	}
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
	}

}
