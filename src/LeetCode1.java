/*
 * 新开一个LeetCode刷题
 * 原来的太多行了
 * */
import java.lang.Math;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Stack;

public class LeetCode1 {

	/*650. 2 Keys Keyboard
	 * 在面板上有一个字符A
	 * 有两个操作：1、复制当前面板上的所有字符，2、当前复制的内容粘贴到面板上
	 * 现给定n，通过1、2操作生成n个字符A
	 * 输出需要的最少操作
	 * 思路因式分解，找到最小的因子
	 * */
	public int minSteps(int n) {
        int i;
        boolean flag = false;
        if(n==1)
        	return 0;
        for(i=2;i<=Math.sqrt(n);i++)
        	if(n%i==0) {
        		flag = true;
        		break;
        	}
        if(flag) {
        	return minSteps(n/i)+i;
        }
		return n;
    }
	
	public int rand7() {
		return (int) (Math.random()*7)+1;
	}
	
	/*470.使用Rand7()实现Rand10()
	 * 使用最少的Rand7()
	 * 思想拒绝采样
	 *    1  2  3  4  5  6  7
	 * 1  1  2  3  4  5  6  7
	 * 2  8  9  10 1  2  3  4
	 * 3  5  6  7  8  9  10 1
	 * 4  2  3  4  5  6  7  8
	 * 5  9  10 1  2  3  4  5
	 * 6  6  7  8  9  10 *  *
	 * 7  *  *  *  *  *  *  *
	 * */
	public int rand10() {
		int idx = 49;
        while(idx>40){
            int row = rand7();
            int col = rand7();
            idx = (row-1)*7 + col;
        }
        return (idx-1)%10 +1;
    }
	
	/* 726. Number of Atoms
	 * 给一个化学公式，计算各个原子数
	 * H2O、H2O2、(H2O2)3、(H2O2)
	 * 输出元素原子数...
	 * 字母序排序
	 * 基本思想栈
	 * */
	public String countOfAtoms(String formula) {
        String result = "";
        int N = formula.length();
        Stack<HashMap<String, Integer>> stack = new Stack<>(); 
		HashMap<String, Integer> top,current;
		int i,num;
		String name;
		int start;
		stack.push(new HashMap<>());
		for(i=0;i<N;) {
			if(formula.charAt(i)=='(') {
				stack.push(new HashMap<>());
				i++;
			}
			else if(formula.charAt(i)==')') {
				top = stack.pop();
				current = stack.peek();
				i++;
				start = i;
				while(i<N && Character.isDigit(formula.charAt(i))) i++;
				num = 1;
				if(i>start) num = Integer.parseInt(formula.substring(start,i));
				for(String str:top.keySet()) {
					if(current.containsKey(str))
						current.put(str, current.get(str)+top.get(str)*num);
					else
						current.put(str, top.get(str)*num);
				}
			} else {
				current = stack.peek();
				start = i;
				i++;
				while(i<N && Character.isLowerCase(formula.charAt(i))) i++;
				name = formula.substring(start,i);
				start = i;
				while(i<N && Character.isDigit(formula.charAt(i))) i++;
				num = 1;
				if(i>start) num = Integer.parseInt(formula.substring(start,i));
				if(current.containsKey(name))
					current.put(name, current.get(name)+num);
				else
					current.put(name, num);
			}
		}
        current = stack.pop();
        String []names = new String[current.size()];
        current.keySet().toArray(names);
        Arrays.sort(names);
        for(String str:names) {
        	result += str;
        	if(current.get(str)>1)
        		result += current.get(str);
        }
        	
		return result;
    }
	
	/* 898. Bitwise ORs of Subarrays
	 * 子数组位或
	 * 数组的所有子数组，子数组元素位或操作不同结果数
	 * 思想：使用HashSet
	 * */
	public int subarrayBitwiseORs(int[] A) {
        HashSet<Integer> cur = new HashSet<>();
        HashSet<Integer> ans = new HashSet<>();
        
        for(int x:A) {
        	HashSet<Integer> cur2 = new HashSet<>();
        	for(int y:cur)
        		cur2.add(x|y);
        	cur2.add(x);
        	cur = cur2;
        	ans.addAll(cur);
        }
		return ans.size();
    }
	
	/* 859. Buddy Strings
	 * 字符串A和B，如果交换A中的两个字母后和B相等，则返回true
	 * */
	public boolean buddyStrings(String A, String B) {
        int i,j=-1,k=0;
        int count[] = new int[26];
        if(A.length()!=B.length())
        	return false;
        int n = A.length();
        if(n==0)
        	return false;
        for(i=0;i<n;i++) {
        	if(A.charAt(i)!=B.charAt(i)) {
        		k=1;
        		if(j==-1)
        			j = i;
        		else if(j==-2) 
        			return false;
        		else {
        			if(A.charAt(i)!=B.charAt(j) || A.charAt(j)!=B.charAt(i))
        				return false;
        		}
        	}
        	if(k==0) {
        		count[A.charAt(i)-'a']++;
            	if(count[A.charAt(i)-'a']==2)
            		k=1;
        	}
        	
        }
        return k==1;
    }
	
	/* 563. Binary Tree Tilt
	 * 返回给定二叉树的Tilt
	 * 二叉树节点的一个tilt是其左子树之和和右子树之和的差的绝对值
	 * 整棵树的tilt是所有节点tilt的和
	 * */
	class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;
		TreeNode(int x) { 
			left = null;
			right = null;
			val = x; }
	}
	public List<Integer> tiltSum(TreeNode root) {
		List<Integer> result = new ArrayList<>();
		if(root==null){
			result.add(0);
			result.add(0);
		}else {
			List<Integer> left = tiltSum(root.left);
			List<Integer> right = tiltSum(root.right);
			result.add(left.get(0)+right.get(0)+root.val);
			int tilt = Math.abs(left.get(0)-right.get(0));
			tilt += left.get(1)+right.get(1);
			result.add(tilt);
		}
		return result;
	}
	public int findTilt(TreeNode root) {
        List<Integer> result = tiltSum(root);
		return result.get(1);
    }
	
	/* 599. Minimum Index Sum of Two Lists
	 * 找到两个列表中相同字符串，且这两个字符串在两个列表中的下标之和最小
	 * */
	public String[] findRestaurant(String[] list1, String[] list2) {
        int n1 = list1.length;
        int n2 = list2.length;
        int n = n1+n2;
        int d,i,j;
        List<String> res = new ArrayList<>();
        for(d=0;d<n-1;d++) {
        	i = 0;
        	if(d>n2-1)
        		i = d-n2+1;
        	for(;i<=d && i<n1;i++) {
        		j = d-i;
        		if(list1[i].equals(list2[j])) {
        			res.add(list1[i]);
        		}
        	}
        	if(!res.isEmpty())
        		break;
        }
        String result[] = new String[res.size()];
        res.toArray(result);
		return result;
    }
	
	/* 830. Positions of Large Groups
	 * 字符串中连续的重复出现3次及以上的称为Large Groups
	 * 找出字符串中所有Large Groups的下标
	 * */
	public List<List<Integer>> largeGroupPositions(String S) {
       List<List<Integer>> res = new ArrayList<>();
       int i,j;
       j = 0;
       for(i=0;i<S.length();i++) {
    	   if(S.charAt(i)!=S.charAt(j)) {
    		   if(i-j>=3) {
    			   List<Integer> tmp = new ArrayList<>();
    			   tmp.add(j);
    			   tmp.add(i-1);
    			   res.add(tmp);
    		   }
    		   j = i;
    	   }
       }
       if(i-j>=3) {
		   List<Integer> tmp = new ArrayList<>();
		   tmp.add(j);
		   tmp.add(i-1);
		   res.add(tmp);
	   }
       return res;
    }
	
	/* 926. Flip String to Monotone Increasing
	 * 将字符串反转为单调增的字符串
	 * 一个包含0和1的字符串，如果是由若干个0后跟随若干个1组成，
	 * 则称这个字符串是单调增的字符串
	 * 输入一个字符串，给出最小的反转次数，
	 * 使得这个字符串是单调增字符串
	 * */
	public int minFlipsMonoIncr(String S) {
		int n = S.length();
        if(n<2)
        	return 0;
		int []a = new int[n];//从左到右计数0
		int []b = new int[n];//从右到左计数1
		int i,min,x;
		if(S.charAt(0)=='0')
			a[0] = 1;
		else
			a[0] = 0;
		if(S.charAt(n-1)=='1')
			b[n-1] = 1;
		else
			b[n-1] = 0;
		for(i=1;i<n;i++) {
			if(S.charAt(i)=='0')
				a[i] = a[i-1]+1;
			else
				a[i] = a[i-1];
			if(S.charAt(n-i-1)=='1')
				b[n-i-1] = b[n-i]+1;
			else
				b[n-i-1] = b[n-i];
		}
		min = n-b[0];
		for(i=1;i<n;i++) {
			x = i-a[i-1]+n-i-b[i];
			if(x<min)
				min = x;
		}
		if(min>n-a[n-1])
			min = n-a[n-1];
		return min;
    }
	
	/*
	 * 9. Palindrome Number
	 * 回文数字
	 * */
	
	boolean isPalindrome(int x) {
        if(x<0 || (x%10==0 && x!=0))
        	return false;
        int revertNum=0;
        while(x>revertNum) {
        	revertNum = revertNum * 10 + x % 10;
        	x /= 10;
        }
        return x==revertNum || x==revertNum/10;
    }
	/*
	 * 6. ZigZag Conversion
	 * 字母按Z形式排列，请将字母按行读出
	 * 
Input: s = "PAYPALISHIRING", numRows = 3
Output: "PAHNAPLSIIGYIR"
Explanation:
P   A   H   N
A P L S I I G
Y   I   R

Input: s = "PAYPALISHIRING", numRows = 4
Output: "PINALSIGYAHRPI"
Explanation:
P     I    N
A   L S  I G
Y A   H R
P     I
	 * */
	
	public String convert(String s, int numRows) {
		int i,j;
		int flag = 1;
		int n = s.length();
		if(n<=numRows || numRows==1)
			return s;
        String result = "";
        int d = (numRows-1)*2;
        int d1 = d;
        for(i=0;i<numRows;i++) {
        	j = i;
        	flag = 1;
        	if(d1==0)
        		d1 = d;
        	while(j<n) {
        		result+= s.charAt(j);
        		if(d1==d)
        			j+= d1;
        		else if(flag==1)
        			j += d1;
        		else 
        			j += (d-d1);
        		flag = -flag;
        	}
        	d1 -= 2;
        }
        
        return result;
    }
	
	/*
	 * 49. Group Anagrams
	 * 将具有字母组成相同的字符串分组
	 * */
	public List<List<String>> groupAnagrams(String[] strs) {
		HashMap<String,List<String>> table = new HashMap<>();
		char[] strc;
		String ss;
		List<String> tmp;
		for(String str:strs) {
			strc = str.toCharArray();
			Arrays.sort(strc);
			ss = new String(strc);
			if(table.containsKey(ss)) {
				tmp= table.get(ss);
				tmp.add(str);
			}else {
				tmp = new ArrayList<>();
				tmp.add(str);
				table.put(ss, tmp);
			}
		}
		List<List<String>> result = new ArrayList<>();
		for(List<String> sp:table.values())
			result.add(sp);
		return result;
    }
	
	/*
	 * 153. Find Minimum in Rotated Sorted Array
	 * 一个升序排列的数组，在某个位置将左右两边调换了
	 * 找到该数组最小元素
	 * 无重复
	 * 思想，折半查找
	 * */
	
	public int findMin(int[] nums) {
        int i=0,j=nums.length-1;
        if(nums[j]>nums[0])
        	return nums[0];
        int mid;
        while(i<j) {
        	mid = (i+j)/2;
        	if(nums[i]>nums[j])
        		i = mid;
        	if(nums[i]<nums[j])
        		j = mid;
        }
        
        if(i==j)
        	return nums[i-1];
        else
        	return nums[i];
    }
	
	/*
	 * 8. String to Integer (atoi)
	 * 字符串转数字
	 * 以非数字开头，返回0
	 * 以数字开头，返回前面的数
	 * */
	
	public int myAtoi(String str) {
		long result = 0;
        int flag = 1;
        int i = 0;
        for(;i<str.length() && str.charAt(i)==' ';i++);
        if(i==str.length())
        	return 0;
        if(str.charAt(i)=='-') {
        	flag = -1;
        	i++;
        } else if (str.charAt(i)=='+') {
        	flag = 1;
        	i++;
        }
        else if(!Character.isDigit(str.charAt(i)))       	
        	return 0;
        
		for(;i<str.length() && Character.isDigit(str.charAt(i));i++) {
			int tmp = str.charAt(i)-'0';
			result = result*10 + tmp;
			if(result>2147483647) {
				if(flag<0)
					return -2147483648;
				else return 2147483647;
			}
				
		}
		
		return (int) (flag*result);
    }
	
	/*
	 * 987. Vertical Order Traversal of a Binary Tree
	 * 二叉树的垂直遍历
	 * */
	public class tuple implements Comparable<tuple>{
		int level;
		int col;
		int val;
		tuple(int a,int b,int c){
			level = a;
			col = b;
			val = c;
		}
		@Override
		public int compareTo(tuple o) {
			// TODO Auto-generated method stub
			if(this.col<o.col)
				return -1;
			if(this.col>o.col)
				return 1;
			if(this.level<o.level)
				return -1;
			if(this.level>o.level)
				return 1;
			if(this.val<o.val)
				return 1;
			if(this.val>o.val)
				return -1;
			return 0;
		}
	}
	
	List<tuple> data = new ArrayList<>();
	
	public void Traversal(TreeNode root,int level,int col) {
		if(root!=null) {
			data.add(new tuple(level, col, root.val));
			Traversal(root.left, level+1, col-1);
			Traversal(root.right, level+1, col+1);
		}
	}
	public List<List<Integer>> verticalTraversal(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();
        Traversal(root, 0, 0);
        tuple all[] = (tuple[])data.toArray(new tuple[data.size()]);
        Arrays.sort(all);
        List<Integer> tmp = new ArrayList<>();
        int i;
        int col = all[0].col;
        result.add(tmp);
        for(i=0;i<all.length;i++) {
        	if(col!=all[i].col) {
        		tmp = new ArrayList<>();
        		result.add(tmp);
        		col = all[i].col;
        	}
        	tmp.add(all[i].val);
        }
        
        return result;
    }
	
	/*
	 * 946. Validate Stack Sequences
	 * 两个数组，如果一个数组能通过push和pop转换成另一个数组，则返回true
	 * 同类思考：二叉树先序遍历序列和后序遍历序列能否组成二叉树
	 * 贪心弹出栈中元素
	 * */
	public boolean validateStackSequences(int[] pushed, int[] popped) {
        Stack<Integer> stack = new Stack<>();
        int i = 0;
        for(int x:pushed) {
        	stack.push(x);
        	while(!stack.isEmpty() && i<popped.length && stack.peek()==popped[i]) {
        		i++;
        		stack.pop();
        	}
        }
		return i==popped.length;
    }
	
	/*
	 * 27. Remove Element
	 * 删除数组中给定值val的元素，并返回剩余数组个数，并且所有剩余元素在最前面
	 * */
	public int removeElement(int[] nums, int val) {
        int i,j;
        j = nums.length;
        for(i=0;i<j;) {
        	if(nums[i]==val) {
        		j--;
        		nums[i] = nums[j];
        	}
        	else
        		i++;
        }
		return j;
    }
	
	/*
	 * 162. Find Peak Element
	 * 找到局部极大值，返回它的下标
	 * 假定nums[-1] = nums[n] = -无穷
	 * 
	 * */
	public int findPeakElement(int[] nums) {
        int i;
        for(i=0;i<nums.length-1;i++)
        	if(nums[i]>nums[i+1])
        		return i;
        return i;
    }
	
	/*
	 * 40. Combination Sum II
	 * 给定一个数组，和一个target数
	 * 给出数组中若干数的和为target的所有组合
	 * */
	
	public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> result = new ArrayList<>();
        Arrays.sort(candidates);
        List<Integer> index = new ArrayList<>();
        int current = candidates[0];
        index.add(0);
        int n = candidates.length;
        int k = 0;
        int i=-1,j;
        while(i<candidates.length) {
        	j = index.get(k)+1;
        	current += candidates[j];
        	if(current>=target) {
        		if(current==target) {
        			
        		}
        		index.remove(k);//移除最后一项
        		k--;
        		j = index.get(k);
        		while(j<n && candidates[j]==candidates[index.get(k)]) j++;
        	}
        	
        	if(current>target) {
        		index.remove(k);//移除最后一项
        		k--;
        		j = index.get(k)+1;
        	}
        }
        
        return result;
    }
	
	
	/*
	 * 15. 3Sum
	 * 数组中找到三个数和为0的组合
	 * 想法，，
	 * 排序，先确定一个数，然后选择两个数，从最小和最大开始
	 * 并去重
	 * */
	public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        int n = nums.length;
        if(n<3)
        	return result;
        Arrays.sort(nums);
        int i,j,k,x;
        i=0;
        while(i<n && nums[i]<=0) {
        	j = i+1;
        	k = n-1;
        	while(j<k)
        		if(nums[i]+nums[j]+nums[k]>0)
        			k--;
        		else if(nums[i]+nums[j]+nums[k]<0)
        			j++;
        		else {
        			List<Integer> tmp = new ArrayList<>();
        			tmp.add(nums[i]);
        			tmp.add(nums[j]);
        			tmp.add(nums[k]);
        			result.add(tmp);
        			x = nums[j];
        			while(j<n && nums[j]==x)
        			j++;
        			k--;
        		}
        	x = nums[i];
        	while(i<n && nums[i]==x) 
        		i++;
        }
        return result;
    }
	
	/*
	 * 16. 3Sum Closest
	 * 找到数组中三个数的和最接近target的数的组合的和
	 * 想法，
	 * 排序，确定一个数，这个数从小到大扫描，必须小于目标数
	 * 然后从最小和最大开始找中间数
	 * */
	
	public int threeSumClosest(int[] nums, int target) {
        int min = Integer.MAX_VALUE;
        int sum = 0;
        int n = nums.length;
        Arrays.sort(nums);
        int i,j,k,x,y;
        i = 0;
        do {
        	j = i+1;
        	k = n-1;
        	while(j<k) {
        		x = nums[i]+nums[j]+nums[k];
        		y = x-target;
        		if(y>0) {
        			k--;
        			if(y<min) {
        				min = y;
        				sum = x;
        			}
        		}
        		else if(y<0) {
        			j++;
        			if(-y<min) {
        				min = -y;
        				sum = x;
        			}
        		} 
        		else return target;
        	}
        	x = nums[i];
        	while(i<n && nums[i]==x) 
        		i++;
        } while(i<n && nums[i]<target);
		return sum;
    }
	
	/*
	 * 18. 4Sum
	 * 找到数组中四个数的和为target的数的组合
	 * 想法：必须遍历有2组合一遍，生成所有两个数组合的和，
	 * 产生一个新的数组，再从这个数组中找到2组合
	 * 排序，
	 * */
	class pairs{
		int x,y;
		public pairs(int a,int b) {
			x = a;y = b;
		}
	}
	public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> result = new ArrayList<>();
        HashMap<Integer,List<pairs>> map = new HashMap<>();
        int i,j,n = nums.length;
        int x,y;
        List<pairs> tmp;
        if(n<4)
        	return result;
        for(i=0;i<n-1;i++)
        	for(j=i+1;j<n;j++) {
        		x = nums[i]+nums[j];
        		if(map.containsKey(x))
        			tmp = map.get(x);
        		else {
        			tmp = new ArrayList<>();
        			map.put(x, tmp);
        		}
        		tmp.add(new pairs(nums[i], nums[j]));
        	}
        Integer twosum[] = (Integer [])map.keySet().toArray();
        
        Arrays.sort(twosum);
        i = 0;
        j = twosum.length-1;
        while(i<j) {
        	x = twosum[i]+twosum[j];
        	if(x>target)
        		j--;
        	else if(x<target)
        		i++;
        	else {
        		
        	}
        }
        return result;
    }
	
	/*
	 * 75. Sort Colors
	 * 思想，和快排中的partion算法类似
	 * */
	public void sortColors(int[] nums) {
        int i,j,k,x;
        j = k = 0;
        for(i=0;i<nums.length;i++) {
        	x = nums[i];
        	nums[i] = 2;
        	if(x<2) {
        		nums[j] = 1;
        		j++;
        	}
        	if(x==0) {
        		nums[k] = 0;
        		k++;
        	}
        }
    }
	
	/*
	 * 3. Longest Substring Without Repeating Characters
	 * 最长，没有重复字符字串
	 * 思想 ：逐个扫描
	 * 方法一：记录当前字串的所有下标，依次从添加
	 * */
	public int lengthOfLongestSubstring(String s) {
		s = s.replaceAll("[^a-zA-Z]", "");
        s = s.toLowerCase();
        if(s.length()==0)
        	return 0;
		int pos[] = new int[26];
		int current,max=0;
		int i,x,j;
		current = 0;
		for(i=0;i<s.length();i++) {
			x = (int)(s.charAt(i) - 'a');
			if(pos[x]==0) {
				pos[x] = i+1; 
				current++;
			}else {
				if(current>max)
					max = current;
				j = i-current;
				current = i-pos[x]+1;
				for(;j<pos[x];j++)
					pos[(int)(s.charAt(i) - 'a')] = 0;
				pos[x] = i+1; 
			}
		}
		if(current>max)
			max = current;
		return max;
    }
	
	/*
	 * 11. Container With Most Water
	 *  给定一个数组，选择两个数，这两个数之间包含最大面积
	 *  思想
	 *  任意两个之间
	 * */
	public int maxArea(int[] height) {
        int n = height.length;
        int i = 0,j = n-1;
        int max = 0,h;
        while(i<j) {
        	if(height[i]<=height[j]) {
        		h = height[i];
        		i++;
        	}else {
        		h = height[j];
        		j--;
        	}
        	if(h*(j-i+1)>max)
        		max = h*(j-i+1);
        }
		return max;
    }
	
	/*
	 * 42.Trapping Rain Water
	 * 给定一个数组，每一个局部极小点可以存储一部分水，
	 * 求数组存水量
	 * 思想，先找到最高处
	 * 从左边和右边分别向最高处遍历
	 * */
	public int trap(int[] height) {
        int i,n = height.length;
        if(n<3)
        	return 0;
        int x,k,s = 0;
        k = 0;
        for(i=1;i<n;i++)
        	if(height[i]>height[k])
        		k = i;
        x = 0;
        for(i=0;i<k;i++)
        	if(x>height[i])
        		s+= (x-height[i]);
        	else
        		x = height[i];
        x = 0;
        for(i=n-1;i>k;i--)
        	if(x>height[i])
        		s+= (x-height[i]);
        	else
        		x = height[i];
		return s;
    }
	
	/*
	 * 	4. Median of Two Sorted Arrays
	 * 	找两个有序数组的中位数
	 * 	想法，每次从两个数组中找到各自的中位数
	 * 	比较两个中位数大小，从而分别保留两个数组各半部分
	 * 	每次舍去的数必须是偶数个
	 * */
	public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        int x1,x2,y1,y2,i,j;
        double m1,m2;
        x1 = x2 = 0;
        y1 = nums1.length;
        y2 = nums2.length;
        while(y1>x1 && y2>x2) {
        	//两个数组都还有数
    		i = (x1+y1)/2;
    		if((y1-x1)%2==0) 
    			m1 = (double)(nums1[i]+nums1[i-1])/2;
    		else
    			m1 = nums1[i];
    		j = (x2+y2)/2;
    		if((y2-x2)%2==0) 
    			m2 = (double)(nums2[j]+nums2[j-1])/2;
    		else
    			m2 = nums2[j];
    		if(m1>m2) {
    			y1 = i;
    			x2 = j;
    		}else if(m1<m2) {
    			x1 = i;
    			y2 = j;
    		}
    		else return m1;
        }
        if(y1>x1) {
        	i = (x1+y1)/2;
    		if((y1-x1)%2==0) 
    			m1 = (double)(nums1[i]+nums1[i-1])/2;
    		else
    			m1 = nums1[i];
    		return m1;
        }
        j = (x2+y2)/2;
		if((y2-x2)%2==0) 
			m2 = (double)(nums2[j]+nums2[j-1])/2;
		else
			m2 = nums2[j];
		return m2;
    }
	
	/*
	 * 14. Longest Common Prefix
	 * 最长公共前缀
	 * */
	public String longestCommonPrefix(String[] strs) {
        int  i,j,m = Integer.MAX_VALUE;
        i = 0;
        if(strs.length==0)
        	return "";
        for(j=0;j<strs.length;j++)
        	if(strs[j].length()<m)
        		m = strs[j].length();
        for(i=0;i<m;i++) {
        	for(j=0;j<strs.length;j++)
        		if(strs[j].charAt(i)!=strs[0].charAt(i)) {
        			return strs[0].substring(0,i);
        		}
        }
        return strs[0].substring(0,i);
    }
	
	/*
	 * 17. Letter Combinations of a Phone Number
	 * 给定一个数串，给出这个数串再9宫格按键中的字母组合
	 * 广度优先搜索
	 * 
	 * */
	public List<String> letterCombinations(String digits) {
        List<String> res = new ArrayList<>();
        List<String> tmp;
		String table[] = {"abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
		if(digits.length()==0)
			return res;
		res.add("");
		for(int i=0;i<digits.length();i++) {
			int k = digits.charAt(i)-'2';
			tmp = new ArrayList<>();
			for(String str:res) {
				for(int j=0;j<table[k].length();j++)
					tmp.add(str+table[k].charAt(j));
			}
			res = tmp;
		}
        return res;
    }
	
	/*
	 * 39.Combination Sum
	 * 给定候选数组集合，无重复，
	 * 找出所有和为target的组合，同一个数字可以重复使用
	 * 思想：分治？动态规划？递归
	 * 回溯？
	 * */
	public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<>();
        if(candidates.length==0)
        	return res;
        Arrays.sort(candidates);
        int i,sum;
        List<Integer> cur = new ArrayList<>();
        sum = 0;;
        cur.add(0);
        while(!cur.isEmpty()) {
        	 i = cur.get(cur.size()-1);
        	 sum += candidates[i];
        	 if(sum<target) {
        		 cur.add(i);
        	 }else {
        		 if(sum==target) {
        			 List<Integer> tmp = new ArrayList<>();
        			 for(int c:cur)
        				 tmp.add(candidates[c]);
        			 res.add(tmp);
        		 }
        		 sum -= candidates[i];
        		 if(i==candidates.length-1)
        			 
        		 cur.remove(cur.size()-1);
        		 
        		 cur.set(cur.size()-1, i+1);
        	 }
        }
        return res;
    }
	
	/*28.Implement strStr()
	 * 字符串匹配，找到needle开始的位置*/
	public int strStr(String haystack, String needle) {
		int i,j,flag;
		if(needle.length()==0)
			return 0;
		if(haystack.length()<needle.length())
			return -1;
		for(i=0;i<haystack.length()-needle.length()+1;i++)
			if(haystack.charAt(i)==needle.charAt(0)) {
				flag = 1;
				for(j=1;j<needle.length();j++)
					if(haystack.charAt(i+j)!=needle.charAt(j)) {
						flag = 0;
						break;
					}
				if(flag==1)
					return i;
			}
		return -1;	
    }
	
	/*
	 * 214.Shortest Palindrome
	 * 给定一个字符串s，在这个字符串前面添加最少的字符
	 * 使得这个字符串变成回文字符串
	 * 思考：找到最长的回文前缀
	 * 问题：怎么最长回文前缀
	 * 方法1：遍历每个前缀，判断前缀是不是回文前缀
	 * n^2时间复杂度？
	 * 方法2：？？？
	 * */
	public String shortestPalindrome(String s) {
        int i;
        String revs = new StringBuffer(s).reverse().toString();
        for(i=0;i<s.length();i++) {
        	if(revs.substring(i).equals(s.substring(0,s.length()-i))) {
        		return revs.substring(0,i)+s;
        	}
        }
		return s;
    }
	
	/*
	 * 336. Palindrome Pairs
	 * 给定若干字符串
	 * 找出所有两个字符串连接后是回文字符串的组合
	 * 遍历所有的组合？
	 * 
	 * */
	public List<List<Integer>> palindromePairs(String[] words) {
		List<List<Integer>> res = new ArrayList<>();
		int i,j;
		List<Integer> tmp;
		for(i=0;i<words.length-1;i++)
			for(j=i+1;j<words.length;j++) {
				String str = words[i] + words[j];
				String rev = new StringBuffer(str).reverse().toString();
				if(str.equals(rev)) {
					tmp = new ArrayList<>();
					tmp.add(i);
					tmp.add(j);
					res.add(tmp);
				}
				str = words[j] + words[i];
				rev = new StringBuffer(str).reverse().toString();
				if(str.equals(rev)) {
					tmp = new ArrayList<>();
					tmp.add(j);
					tmp.add(i);
					res.add(tmp);
				}
			}
		return res;
	}
	
	/*
	 * 443.String Compression
	 * 给定一个字符数组，按恰当的方式压缩
	 * 返回压缩后的长度，压缩后字符保存在chars中
	 * 思考，只有O*/
	public int compress(char[] chars) {
        if(chars.length<2)
        	return chars.length;
        int i,count,j = 0;
        char ch;
        ch = chars[0];
        count = 1;
        for(i=1;i<chars.length;i++) {
        	if(chars[i]==ch)
        		count++;
        	else {
        		chars[j] = ch;
        		j++;
        		if(count>=100) {
        			chars[j] =  (char)(count/100 + '0');
        			chars[j+1] =  (char)(count/10%10 + '0');
        			chars[j+2] =  (char)(count%10 + '0');
        			j+= 3;
        		}else if(count>=10) {
        			chars[j] =  (char)(count/10 + '0');
        			chars[j+1] =  (char)(count%10 + '0');
        			j += 2;
        		}else if(count>1) {
        			chars[j] = (char)(count + '0');
        			j++;
        		}
        		count = 1;
        		ch = chars[i];
        	}
        }
        chars[j] = ch;
		j++;
		if(count>=100) {
			chars[j] =  (char)(count/100 + '0');
			chars[j+1] =  (char)(count/10%10 + '0');
			chars[j+2] =  (char)(count%10 + '0');
			j+= 3;
		}else if(count>=10) {
			chars[j] =  (char)(count/10 + '0');
			chars[j+1] =  (char)(count%10 + '0');
			j += 2;
		}else if(count>1) {
			chars[j] = (char)(count + '0');
			j++;
		}
        return j;
    }
	
	/*
	 * 38.Count and Say
	 * 
1.     1
2.     11
3.     21
4.     1211
5.     111221
1 is read off as "one 1" or 11.
11 is read off as "two 1s" or 21.
21 is read off as "one 2, then one 1" or 1211.
	 */
	public String countAndSay(int n) {
        String res = "";
        String next;
        if(n==0)
        	return res;
        int i,j,count;
        char ch;
        res = "1";
        for(i=1;i<n;i++) {
        	ch = res.charAt(0);
        	count = 1;
        	next = "";
        	for(j=1;j<res.length();j++) {
        		if(res.charAt(j)==ch)
        			count++;
        		else {
        			next += String.valueOf(count);
        			next += ch;
        			ch = res.charAt(j);
                	count = 1;
        
        		
        		}
        	}
        	next += String.valueOf(count);
			next += ch;
        	res = next;
        }
        return res;
    }
	
	/*
	 * 44.Wildcard Maching
	 * 匹配
	 * 给定输入串S，匹配串p
	 * ？匹配单个字符，*匹配任意长度字符，判断是否匹配
	 * 思考：需要对每种情况进行遍历
	 * 对*所在的位置进行匹配
	 * */
	public boolean isMatch(String s, String p) {
        int i,j,k;
        if(p.length()==0 && s.length()==0)
        	return true;
        if(p.length()==0)
        	return false;
        List<Integer> any = new ArrayList<>();
        String np = "";
        char ch = p.charAt(0);
        
        //第一步，移除多余的*号
        
        for(i=0;i<p.length();i++)
        	if(p.charAt(i)=='*')
        		any.add(i);
        int many[] = new int[any.size()];
        
		
        
		
		return true;
    }
	
	/*
	 * 32. Longest Valid Parentheses
	 * 最长匹配问题：给一个括号字符串，找多最长的合法字串
	 * 思想：贪心？分治？动态规划？
	 * */
	public int longestValidParentheses(String s) {
        int x = 0;
        int max = 0;
        Stack<Character> stack = new Stack<>();
        int i;
        for(i=0;i<s.length();i++) {
        	if(s.charAt(i)=='(')
        		stack.push('(');
        	else {
        		if(!stack.isEmpty()) {
        			x+=2;
        			stack.pop();
        		}
        	}
        }
        return x;
    }
	
	/*
	 * 730. Count Different Palindromic Subsequences
	 * 计算字符串中，所有非空回文子串的数目
	 * 返回的结果对10^9+7 取余
	 * 只有a，b，c，d四个字符
	 * */
	public int countPalindromicSubsequences(String S) {
		if(S.length()<2)
			return S.length();
        return 0;
    }
	
	class ListNode {
		int val;
		ListNode next;
		public ListNode(int val) {
			this.val = val;
			next = null;
		}
	}
	/*
	 * 23.Merge k Sorted Lists
	 * k路归并
	 * 最好的方法，堆，
	 * */
	
	class Node{
		int val;
		int p;
		Node left,right;
		public Node(int val,int p) {
			this.val = val;
			this.p = p;
    		left = null;
    		right = null;
		}
	}
	void adjustHeap(int x,Node heap) {
		Node t = heap;
		
	}
	public ListNode mergeKLists(ListNode[] lists) {
        ListNode res = new ListNode(0);
        
        class TreeNode{
        	int val;
        	TreeNode left,right;
        	public TreeNode(int val) {
        		this.val = val;
        		left = null;
        		right = null;
			}
        }
        
        return res;
    }
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		LeetCode1 test = new LeetCode1();
		//"mississippi"
		//"issipi"
		System.out.println(test.longestValidParentheses(")()())"));
		int A[] = {2,3};
		char chars[] = {'o','o','o','o','o','o','o','o','o','o'};
		System.out.println(test.countAndSay(4));
		System.out.println(test.compress(chars));
		for(char sss:chars)
			System.out.println(sss);
		
		System.out.println(test.shortestPalindrome("aacecaaa"));
		System.out.println(test.strStr("mm", "mm"));
		System.out.println(test.combinationSum(A, 5));
		int nums1[] = {5,7};
		int nums2[] = {2};
		String strs[] = {"c","c"};
		System.out.println(test.longestCommonPrefix(strs));
		System.out.println(test.findMedianSortedArrays(nums1, nums2));
		//1,1,2
		//4,5,6,7,0,1,2
		//3,4,5,1,2
		//2,1
		//3,1,2
		System.out.println(test.trap(A));
		System.out.println(test.lengthOfLongestSubstring("hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789hijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"));
		
		int B[] = {3,1,2};
		String sstr = "-91283472332";
		//"-91283472332"
		String []a = {"aac","bab","cca","KFC"};
		String []b = {"ade","dea","cca","KFC"};
		//String []strs = {"eat","tea","tan","ate","nat","bat"};
		TreeNode tt = test.new TreeNode(7);
		TreeNode root = test.new TreeNode(20);
		root.right = tt;
		tt = test.new TreeNode(15);
		root.left = tt;
		tt = test.new TreeNode(3);
		tt.right = root;
		root = test.new TreeNode(9);
		tt.left = root;
		
		int C[] = {2,0,2,1,1,0};
		test.sortColors(C);
		for(int i:C)
			System.out.print(i+"  ");
		
		//int nums[] = {0,0,0};
		//int nums[] = {-2,0,0,2,2};
		int nums[] = {0,1,2};
		int target = 0;
		System.out.println(test.threeSumClosest(nums, target));
		
		
		
		List<List<Integer>> resList;
		
		//resList = test.verticalTraversal(tt);
		resList = test.threeSum(nums);
		for(List<Integer> tmp:resList) {
			for(Integer c:tmp)
				System.out.print(c+" ");
			System.out.println();
		}
			
		System.out.println(test.myAtoi(sstr));
		
		
		System.out.println(test.findMin(B));
		List<List<String>> result = test.groupAnagrams(strs);
		for(List<String> tmp:result) {
			for(String str:tmp)
				System.out.print(str);
			System.out.println();
		}
		
		System.out.println(test.convert("PAYPALISHIRING", 4));
		System.out.println(test.isPalindrome(10));
		System.out.println(test.minFlipsMonoIncr("00011000"));
		System.out.println(test.findRestaurant(a, b)[0]);
		System.out.println(test.buddyStrings("ab","ba"));
		System.out.println(test.subarrayBitwiseORs(A));
		//System.out.println("Hello".substring(0,-1));
		System.out.println(test.minSteps(12));
		System.out.println(test.rand7());
		System.out.println(test.rand10());
		System.out.println(test.countOfAtoms("K4(ON(SO3)2)2"));
		
		
	}

}
