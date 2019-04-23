import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedList;
import java.util.PriorityQueue;
import java.util.Queue;
import java.util.Stack;

public class LeetCode2 {
	/*
	 * 这里是所有剑指offer的题目总结
	 */
	class ListNode {
		int val;
		ListNode next;

		public ListNode(int val) {
			this.val = val;
			next = null;
		}
	}

	class TreeNode {
		int val;
		TreeNode left;
		TreeNode right;

		TreeNode(int x) {
			left = null;
			right = null;
			val = x;
		}
	}

	/*
	 * 1.二进制中1的个数， 思路不断除以2，mod 2 注意，这仍然不是最快的方法 对于32位整型有计数方法，64位也有，这个指令仅限于C++
	 * SSE4指令集，popcnt _mm_popcnt_u32
	 */
	public int NumberOf1(int n) {
		/*
		 * // 简单的思路 int count = 0; while(n>0) { count += n%2;//这里不需要自己做优化，编译器用的就是位移运算 n
		 * /= 2;//请勿自作聪明 } return count;
		 */
		// 方法二,注意方法2在本质上和方法一的时间效率是一样的
		int count = 0;
		while (n != 0) {
			n = n & (n - 1);// 作用，把n最右边的1变0
			count++;
		}
		return count;
	}

	/*
	 * 2.判断二进制中0的个数 每次除以2，mod 2
	 */
	public int findZero(int n) {
		int count = 0;
		while (n != 0) {
			count += (1 - n % 2);// 注意，用取余没有问题
			n /= 2;
		}
		return count;
	}

	/*
	 * 3.二进制高位连续0的个数 方法，每次与最高位为1的二进制进行&操作
	 */
	public int numberOfLeadingZeros0(int n) {
		if (n == 0)
			return 32;
		int count = 0;
		int mask = 0x80000000;
		while ((n & mask) == 0) {
			count++;
			n <<= 1;
		}
		return count;
	}

	/*
	 * 4. 在一个二维数组中（每个一维数组的长度相同）， 每一行都按照从左到右递增的顺序排序， 每一列都按照从上到下递增的顺序排序
	 * 思想：折半查找，先找到所在行，再找所在列 思路：从左上角开始，若小则删除一行，若大，则删除一列 要求，这个数组行列相同
	 */
	public boolean Find(int target, int[][] array) {
		int row = 0;
		int col = array[0].length - 1;
		while (row < array.length && col >= 0) {
			if (array[row][col] > target)
				col--;
			else if (array[row][col] < target)
				row++;
			else
				return true;
		}

		return false;
	}

	/*
	 * 5.将字符串中的所有空格转换成%20 思想，新建一个新的串，依次插入
	 */
	public String replaceSpace(StringBuffer str) {
		String tmp = "";
		int i;
		for (i = 0; i < str.length(); i++) {
			if (str.charAt(i) == ' ')
				tmp += "%20";
			else
				tmp += str.charAt(i);
		}
		return tmp;
	}

	/*
	 * 6..输入一个链表，按链表值从尾到头的顺序返回一个ArrayList 使用栈
	 */
	public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {
		Stack<Integer> stack = new Stack<>();
		ArrayList<Integer> result = new ArrayList<>();
		if (listNode == null)
			return result;
		while (listNode != null) {
			stack.push(listNode.val);
			listNode = listNode.next;
		}
		while (!stack.isEmpty())
			result.add(stack.pop());
		return result;
	}

	/*
	 * 7.把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。 输入一个非减排序的数组的一个旋转， 输出旋转数组的最小元素。
	 * 例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转， 该数组的最小值为1 二分查找
	 */
	public int minNumberInRotateArray(int[] array) {
		int n = array.length;
		if (n == 0)
			return 0;
		if (array[n - 1] > array[0])
			return array[0];
		int mid, i, j;
		i = 0;
		j = n - 1;
		while (i < j) {
			if (j - i == 1)
				break;
			mid = (i + j) / 2;
			if (array[mid] >= array[i]) {
				i = mid;
			} else
				j = mid;
		}
		return array[j];
	}

	/*
	 * 8.大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项
	 */
	public int Fibonacci(int n) {
		if (n == 0)
			return 0;
		else if (n == 1)
			return 1;
		int a, b, t;
		a = 0;
		b = 1;
		for (int i = 2; i <= n; i++) {
			t = a + b;
			a = b;
			b = t;
		}
		return b;
	}

	/*
	 * 9.一只青蛙一次可以跳上1级台阶， 也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法 思想：递归 同斐波那契数列数列
	 */
	public int JumpFloor(int target) {
		if (target == 0)
			return 1;
		else if (target == 1)
			return 1;
		int a, b, t;
		a = 0;
		b = 1;
		for (int i = 0; i < target; i++) {
			t = a + b;
			a = b;
			b = t;
		}
		return b;
	}

	/*
	 * 10.一只青蛙一次可以跳上1级台阶， 也可以跳上2级……它也可以跳上n级。 求该青蛙跳上一个n级的台阶总共有多少种跳法， 指数级 2^n
	 */
	public int JumpFloorII(int target) {
		int n = 1, i;
		for (i = 1; i < target; i++)
			n *= 2;
		return n;
	}

	/*
	 * 11.我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。 请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形， 总共有多少种方法 同斐波那契数列
	 */
	public int RectCover(int target) {
		if (target == 0)
			return 0;
		else if (target == 1)
			return 1;
		if (target == 0)
			return 1;
		else if (target == 1)
			return 1;
		int a, b, t;
		a = 0;
		b = 1;
		for (int i = 0; i < target; i++) {
			t = a + b;
			a = b;
			b = t;
		}
		return b;

	}

	/*
	 * 13.给定一个double类型的浮点数base和int类型的整数exponent。 求base的exponent次方。 快速幂算法
	 */
	public double Power(double base, int exponent) {
		if (exponent == 0)
			return 1;
		else {
			int flag = 1;
			if (exponent < 0) {
				exponent = -exponent;
				flag = -1;
			}
			double x = Power(base, exponent / 2);
			if (exponent % 2 == 0)
				x = x * x;
			else
				x = x * x * base;
			if (flag == -1)
				return 1 / x;
			return x;
		}
	}

	/*
	 * 14. 输入一个整数数组， 实现一个函数来调整该数组中数字的顺序， 使得所有的奇数位于数组的前半部分， 所有的偶数位于数组的后半部分，
	 * 并保证奇数和奇数，偶数和偶数之间的相对位置不变。 时间复杂度：基于交换n^2 每次从下一个位置拿出一个数 方法二，新数组
	 */
	public void reOrderArray(int[] array) {
		int n = array.length;
		int i, j, k;
		if (n == 0)
			return;
		int tmp[] = new int[n];
		for (i = 0, j = 0; i < n; i++)
			if (array[i] % 2 == 1) {
				tmp[j] = array[i];
				j++;
			}
		for (i = n - 1, k = n - 1; i >= 0; i--) {
			if (array[i] % 2 == 0) {
				array[k] = array[i];
				k--;
			}
		}
		for (i = 0; i < j; i++)
			array[i] = tmp[i];
	}

	/*
	 * 15.输入一个链表，输出该链表中倒数第k个结点 思想，一个指针在前，一个在后
	 */
	public ListNode FindKthToTail(ListNode head, int k) {
		ListNode res;
		ListNode pre;
		int i;
		pre = head;
		for (i = 0; i < k; i++)
			if (pre != null)
				pre = pre.next;
			else
				return null;
		res = head;
		while (pre != null) {
			pre = pre.next;
			res = res.next;
		}
		return res;
	}

	/*
	 * 16.输入一个链表，反转链表后，输出新链表的表头 头插法
	 */
	public ListNode ReverseList(ListNode head) {
		if (head == null)
			return null;
		ListNode res = head;
		head = head.next;
		res.next = null;
		ListNode t;
		while (head != null) {
			t = head;
			head = head.next;
			t.next = res;
			res = t;
		}
		return res;
	}

	/*
	 * 16.链表归并 输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则
	 */
	public ListNode Merge(ListNode list1, ListNode list2) {
		if (list1 == null)
			return list2;
		if (list2 == null)
			return list1;
		ListNode head = new ListNode(-1);
		ListNode tail = head;
		while (list1 != null && list2 != null) {
			if (list1.val < list2.val) {
				tail.next = list1;
				tail = tail.next;
				list1 = list1.next;
			} else {
				tail.next = list2;
				tail = tail.next;
				list2 = list2.next;
			}
		}
		if (list1 == null)
			tail.next = list2;
		else
			tail.next = list1;
		return head.next;
	}

	/*
	 * 17.输入两棵二叉树A，B，判断B是不是A的子结构。 考点递归
	 */
	public boolean isSubtree(TreeNode root1, TreeNode root2) {
		// 判断从根节点开始
		if (root2 == null)
			return true;
		if (root1 == null)
			return false;
		if (root1.val != root2.val)
			return false;
		return isSubtree(root1.left, root2.left) && isSubtree(root1.right, root2.right);
	}

	public boolean HasSubtree(TreeNode root1, TreeNode root2) {
		if (root1 == null || root2 == null)
			return false;
		boolean res = false;
		if (root1.val == root2.val)
			res = isSubtree(root1, root2);
		res = res || HasSubtree(root1.left, root2) || HasSubtree(root1.right, root2);
		return res;
	}

	/*
	 * 18.镜像二叉树 操作给定的二叉树，将其变换为源二叉树的镜像。 考点：递归
	 */
	public void Mirror(TreeNode root) {
		if (root != null) {
			TreeNode left = root.left;
			TreeNode right = root.right;
			root.right = left;
			root.left = right;
			Mirror(root.left);
			Mirror(root.right);
		}
	}

	/*
	 * 19.顺时针输出二维数组 输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字
	 */
	public ArrayList<Integer> printMatrix(int[][] matrix) {
		ArrayList<Integer> res = new ArrayList<>();
		int m = matrix.length;
		if (m == 0)
			return res;
		int n = matrix[0].length;
		int i, j, k = 0, t;
		i = 0;
		j = -1;
		while (m > 0 && n > 0) {
			if (k == 0) {
				for (t = 0; t < n; t++) {
					j++;
					res.add(matrix[i][j]);
				}
				m--;
			} else if (k == 1) {
				for (t = 0; t < m; t++) {
					i++;
					res.add(matrix[i][j]);

				}
				n--;
			} else if (k == 2) {
				for (t = 0; t < n; t++) {
					j--;
					res.add(matrix[i][j]);

				}
				m--;
			} else if (k == 3) {
				for (t = 0; t < m; t++) {
					i--;
					res.add(matrix[i][j]);
				}
				n--;
			}
			k = (k + 1) % 4;
		}
		return res;
	}

	/*
	 * 20.栈 能获取栈中最小元素
	 */
	Stack<Integer> data = new Stack<>();
	Integer[] sorted;

	public void push1(int node) {
		data.push(node);
		sorted = (Integer[]) data.toArray(new Integer[data.size()]);
		Arrays.sort(sorted);
	}

	public void pop1() {
		data.pop();
		sorted = (Integer[]) data.toArray(new Integer[data.size()]);
		Arrays.sort(sorted);
	}

	public int top() {
		int x = data.pop();
		data.push(x);
		return x;
	}

	public int min() {
		return sorted[0];
	}

	/*
	 * 21.判断一个序列能否通过栈转变成另一个序列 输入两个整数序列，第一个序列表示栈的压入顺序， 请判断第二个序列是否可能为该栈的弹出顺序。
	 * 1,2,3,4,5是某栈的压入顺序 序列4,5,3,2,1是该压栈序列对应的一个弹出序列 但4,3,5,1,2就不可能是该压栈序列的弹出序列 思想，贪心
	 */
	public boolean IsPopOrder(int[] pushA, int[] popA) {
		Stack<Integer> stack = new Stack<>();
		int i = 0;
		for (int x : pushA) {
			stack.push(x);
			while (!stack.isEmpty() && i < popA.length && stack.peek() == popA[i]) {
				stack.pop();
				i++;
			}
		}
		return i == popA.length;
	}

	/*
	 * 22.从上往下打印出二叉树的每个节点，同层节点从左至右打印。 思想：广度优先搜索
	 */
	public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {
		ArrayList<Integer> res = new ArrayList<>();
		ArrayList<TreeNode> cur, next;
		if (root == null)
			return res;
		cur = new ArrayList<>();
		cur.add(root);
		while (cur.size() != 0) {
			next = new ArrayList<>();
			for (TreeNode x : cur) {
				res.add(x.val);
				if (x.left != null)
					next.add(x.left);
				if (x.right != null)
					next.add(x.right);
			}
			cur = next;
		}
		return res;
	}

	/*
	 * 23.输入一个整数数组， 判断该数组是不是某二叉搜索树的后序遍历的结果。 如果是则输出Yes,否则输出No。 假设输入的数组的任意两个数字都互不相同。
	 * 考点：搜索二叉树（左子树大根节点，右子树小根节点） 思想：递归，从最左侧开始 最右为根节点，剩余的节点左边小，右边大
	 */
	public boolean VerifySquenceOfBST(int[] sequence) {
		int n = sequence.length;
		if (n == 0)
			return false;
		if (n < 3)
			return true;
		int i, j;
		for (i = 0; i < n; i++)
			if (sequence[i] > sequence[n - 1])
				break;
		for (j = i + 1; j < n; j++)
			if (sequence[j] < sequence[n - 1])
				return false;
		boolean res = true;
		if (i > 1)
			res = res && VerifySquenceOfBST(Arrays.copyOfRange(sequence, 0, i - 1));
		if (i < n - 1)
			res = res && VerifySquenceOfBST(Arrays.copyOfRange(sequence, i, n - 1));
		return res;
	}

	/*
	 * 24.输入一颗二叉树的跟节点和一个整数， 打印出二叉树中结点值的和为输入整数的所有路径。
	 * 路径定义为从树的根结点开始往下一直到叶结点所经过的结点形成一条路径。 (注意: 在返回值的list中，数组长度大的数组靠前) 思考：递归
	 */
	public ArrayList<ArrayList<Integer>> FindPath(TreeNode root, int target) {
		ArrayList<ArrayList<Integer>> res = new ArrayList<>();
		ArrayList<ArrayList<Integer>> left;
		ArrayList<ArrayList<Integer>> right;
		ArrayList<Integer> path;
		if (root == null)
			return res;
		if (root.val == target && root.left == null && root.right == null) {
			path = new ArrayList<>();
			path.add(root.val);
			res.add(path);
			return res;
		}
		if (root.left == null && root.right == null)
			return res;
		left = FindPath(root.left, target - root.val);
		right = FindPath(root.right, target - root.val);
		int i, j;
		i = 0;
		j = 0;
		while (i < left.size() && j < right.size()) {
			path = new ArrayList<>();
			path.add(root.val);
			res.add(path);
			if (left.get(i).size() > right.get(i).size()) {
				path.addAll(left.get(i));
				i++;
			} else {
				path.addAll(right.get(j));
				j++;
			}
		}
		if (i == left.size()) {
			for (; j < right.size(); j++) {
				path = new ArrayList<>();
				path.add(root.val);
				res.add(path);
				path.addAll(right.get(j));
			}
		} else {
			for (; i < left.size(); i++) {
				path = new ArrayList<>();
				path.add(root.val);
				res.add(path);
				path.addAll(left.get(i));
			}
		}
		return res;
	}

	public class RandomListNode {
		int label;
		RandomListNode next = null;
		RandomListNode random = null;

		public RandomListNode(int label) {
			this.label = label;
		}
	}

	/*
	 * 25. 输入一个复杂链表（每个节点中有节点值， 以及两个指针，一个指向下一个节点， 另一个特殊指针指向任意一个节点），
	 * 返回结果为复制后复杂链表的head。 方法一：先建立顺序的链表， 然后对每个节点，查找Random指针的位置，n^2
	 * 方法二：使用hash表，扫描两次，分别得到各个节点的下标
	 */
	public RandomListNode Clone(RandomListNode pHead) {
		RandomListNode p;
		HashMap<RandomListNode, Integer> table = new HashMap<>();
		p = pHead;
		if (pHead == null)
			return null;
		int i;
		i = 0;
		while (p != null) {
			table.put(p, i);
			i++;
			p = p.next;
		}
		int j;
		RandomListNode data[] = new RandomListNode[i];
		p = pHead;
		for (j = 0; j < i; j++) {
			data[j] = new RandomListNode(p.label);
			p = p.next;
		}
		p = pHead;
		for (j = 0; j < i - 1; j++) {
			data[j].next = data[j + 1];
			if (p.random != null)
				data[j].random = data[table.get(p.random)];
			p = p.next;
		}
		if (p.random != null)
			data[i - 1].random = data[table.get(p.random)];
		return data[0];

	}

	/*
	 * 26.输入一棵二叉搜索树， 将该二叉搜索树转换成一个排序的双向链表。 要求不能创建任何新的结点，只能调整树中结点指针的指向 思考：递归
	 * 
	 */
	public TreeNode Convert(TreeNode pRootOfTree) {
		TreeNode left, right;
		if (pRootOfTree == null)
			return null;
		if (pRootOfTree.left != null) {
			left = Convert(pRootOfTree.left);
			while (left.right != null)
				left = left.right;
			left.right = pRootOfTree;
			pRootOfTree.left = left;
		}
		if (pRootOfTree.right != null) {
			right = Convert(pRootOfTree.right);
			while (right.left != null)
				right = right.left;
			right.left = pRootOfTree;
			pRootOfTree.right = right;
		}
		while (pRootOfTree.left != null)
			pRootOfTree = pRootOfTree.left;
		return pRootOfTree;
	}

	/*
	 * 27.字符串的字典序全排列 思想：递归，a+bcd,b+acd,c+abd,
	 */
	public ArrayList<String> Permutation(String str) {
		ArrayList<String> res = new ArrayList<>();
		if (str.length() == 0)
			return res;
		if (str.length() == 1) {
			res.add(str);
			return res;
		}
		char chs[] = str.toCharArray();
		Arrays.sort(chs);
		char ch;
		int i, j;
		ArrayList<String> tmp;
		i = 0;
		j = 0;
		while (i < chs.length) {
			ch = chs[i];
			chs[i] = chs[0];
			String s = new String(chs);
			tmp = Permutation(s.substring(1));
			for (String x : tmp)
				res.add(ch + x);
			chs[i] = ch;
			while (j < chs.length && chs[j] == chs[i])
				j++;
			i = j;
		}
		return res;
	}

	/*
	 * 28.数组中有个数字出现次数超过数组长度的一半 如果不存在返回0 排序，Hashmap
	 */
	public int MoreThanHalfNum_Solution(int[] array) {
		HashMap<Integer, Integer> table = new HashMap<>();
		if (array.length == 1)
			return array[0];
		for (int x : array) {
			if (!table.containsKey(x)) {
				table.put(x, 1);
			} else {
				int t = table.get(x);
				if (t >= array.length / 2)
					return x;
				table.put(x, t + 1);
			}
		}
		return 0;
	}

	/*
	 * 29.输入n个整数，找出其中最小的K个数 思想：分治，快排
	 */
	public ArrayList<Integer> GetLeastNumbers_Solution(int[] input, int k) {
		ArrayList<Integer> below = new ArrayList<>();
		int i, j, x, n = input.length;
		int left, right, key;
		if (n < k)
			return below;
		left = 0;
		right = n - 1;
		while (true) {
			x = (int) (left + Math.random() * (right - left + 1));
			key = input[x];
			input[x] = input[left];
			i = left;
			j = right;
			while (i < j) {
				while (i < j && input[j] >= key)
					j--;
				input[i] = input[j];
				while (i < j && input[i] <= key)
					i++;
				input[j] = input[i];
			}
			input[i] = key;
			if (i == k || i + 1 == k) {
				for (j = 0; j < k; j++)
					below.add(input[j]);
				return below;
			}
			if (i < k)
				left = i;
			else
				right = i;
		}
	}

	/*
	 * 30.连续数组最大和 思想：分治？动态规划 思想是动态规划，如果加上当前位置的数后，和比当前位置的数小，则重新开始
	 */
	public int FindGreatestSumOfSubArray(int[] array) {
		int i;
		int maxsum = array[0];
		int cursum = array[0];
		for (i = 1; i < array.length; i++) {
			cursum += array[i];
			if (cursum < array[i])
				cursum = array[i];
			if (cursum > maxsum)
				maxsum = cursum;
		}
		return maxsum;
	}

	/* 31.二叉树的深度 */
	public int TreeDepth(TreeNode root) {
		if (root == null)
			return 0;
		else {
			int x = TreeDepth(root.left);
			int y = TreeDepth(root.right);
			if (x >= y)
				return x + 1;
			else
				return y + 1;
		}
	}

	/*
	 * 32.一个整型数组里，除两个数字外，其它数字都出现了偶数次 使用xor，异或的性质 a^a = 0; 0^a = a;
	 */
	public void FindNumsAppearOnce(int[] array, int num1[], int num2[]) {
		if (array.length < 2)
			return;
		int myxor = 0;
		int flag = 1;
		for (int i = 0; i < array.length; ++i)
			myxor ^= array[i];
		while ((myxor & flag) == 0)
			flag <<= 1;// 用于
		// num1[0] = myxor;
		// num2[0] = myxor;
		for (int i = 0; i < array.length; ++i) {
			if ((flag & array[i]) == 0)
				num2[0] ^= array[i];
			else
				num1[0] ^= array[i];
		}
	}

	/*
	 * 33.和为S的连续正数序列 首先，（x-y+1)*(x+y)/2 = S 2*S = a*b;x-y+1 = a;x+y = b; 2x =
	 * a+b-1;2y = b-a+1 求，a，b同号的所有因子
	 */
	public ArrayList<ArrayList<Integer>> FindContinuousSequence(int sum) {
		int i, j, k, x, y;
		ArrayList<ArrayList<Integer>> res = new ArrayList<>();
		sum *= 2;
		for (i = (int) Math.sqrt(sum); i >= 1; i--)
			if (sum % i == 0) {
				j = sum / i;
				x = j - i + 1;
				y = j + i - 1;
				if (x % 2 == 0 && x < y) {
					ArrayList<Integer> tmp = new ArrayList<>();
					res.add(tmp);
					for (k = x / 2; k <= y / 2; k++)
						tmp.add(k);
				}
			}
		return res;
	}

	/*
	 * 34.输入一个递增排序的数组和一个数字S， 在数组中查找两个数，使得他们的和正好是S， 如果有多对数字的和等于S， 输出两个数的乘积最小的。
	 * 思考：从两边开始找
	 * 
	 */
	public ArrayList<Integer> FindNumbersWithSum(int[] array, int sum) {
		int i, j;
		ArrayList<Integer> res = new ArrayList<>();
		i = 0;
		j = array.length - 1;
		while (i < j) {
			int k = array[i] + array[j];
			if (k > sum)
				j--;
			else if (k < sum)
				i++;
			else {
				res.add(array[i]);
				res.add(array[j]);
				break;
			}
		}
		return res;
	}

	/*
	 * 35.将给定字符串循环左移n位 问题在于左移位数超过字符串长度
	 */
	public String LeftRotateString(String str, int n) {
		int len = str.length();
		if (len == 0)
			return str;
		n = n % len;
		return str.substring(n, len) + str.substring(0, n);
	}

	/*
	 * 36.将字符串中的单词逆序 “student. a am I” “I am a student.”
	 */
	public String ReverseSentence(String str) {
		String res = "";
		int i, j;
		i = 0;
		j = 0;
		while (j < str.length()) {
			if (str.charAt(j) == ' ') {
				res = " " + str.substring(i, j) + res;
				i = j + 1;
			}
			j++;
		}
		res = str.substring(i, j) + res;
		return res;
	}

	/*
	 * 37. LL今天心情特别好, 因为他去买了一副扑克牌,发现里面居然有2个大王, 2个小王(一副牌原本是54张^_^)...他随机从中抽出了5张牌,
	 * 想测测自己的手气,看看能不能抽到顺子,如果抽到的话,他决定去买体育彩票,嘿嘿！！ “红心A,黑桃3,小王,大王,方片5”,“Oh My
	 * God!”不是顺子.....LL不高兴了, 他想了想,决定大\小 王可以看成任何数字,并且A看作1,J为11,Q为12,K为13。
	 * 上面的5张牌就可以变成“1,2,3,4,5”(大小王分别看作2和4),“So Lucky!”。 LL决定去买体育彩票啦。
	 * 现在,要求你使用这幅牌模拟上面的过程, 然后告诉我们LL的运气如何， 如果牌能组成顺子就输出true，否则就输出false。
	 * 为了方便起见,你可以认为大小王是0。
	 */
	public boolean isContinuous(int[] numbers) {
		Arrays.sort(numbers);
		int i;
		if (numbers.length != 5)
			return false;
		int s = 0;
		int count = 0;
		for (i = 0; i < 5; i++)
			if (numbers[i] == 0)
				count++;
			else
				break;
		i++;
		for (; i < 5; i++)
			s += numbers[i] - numbers[i - 1];
		System.out.println(s + "S" + count);
		if (s <= 4 && s + count >= 4)
			return true;
		return false;
	}

	/*
	 * 38.猴子报数问题 约瑟夫问题 通过递归公式 f(n) = f(n-1)+m mod n 方法，从n开始往n-1看
	 */
	public int LastRemaining_Solution(int n, int m) {
		if (n == 0 && m == 0)
			return -1;
		int s = 0, i;
		for (i = 2; i <= n; i++)
			s = (s + m) % i;
		return s;
	}

	/*
	 * 39.求1+2+3+...+n， 要求不能使用乘除法、for、while、if、else、switch、case 等关键字及条件判断语句（A?B:C）。
	 * 思路：使用逻辑与的短路特性
	 */
	public int Sum_Solution(int n) {
		boolean ans = (n > 0) && (n += Sum_Solution(n - 1)) > 0;
		return n;
	}

	/*
	 * 40.字符出转换成整数 包括数字字母符号,可以为空 数值为0或者字符串不是一个合法的数值则返回0
	 */
	public int StrToInt(String str) {
		int s = 0;
		int i = 0;
		int flag = 1;
		if (str.length() == 0)
			return 0;
		if (str.charAt(0) == '+')
			i = 1;
		else if (str.charAt(0) == '-') {
			i = 1;
			flag = -1;
		}
		for (; i < str.length(); i++) {
			char ch = str.charAt(i);
			if ('0' <= ch && ch <= '9') {
				s = s * 10 + (int) (ch - '0');
			} else
				return 0;
		}
		return s * flag;
	}

	/*
	 * 41.写一个函数， 求两个整数之和， 要求在函数体内不得使用+、-、*、/四则运算符号。 使用位运算
	 */
	public int Add(int num1, int num2) {
		while (num2 != 0) {
			int temp = num1 ^ num2;
			num2 = (num1 & num2) << 1;
			num1 = temp;
		}
		return num1;
	}

	/*
	 * 42.找到数组中的一个重复的数 数组中有若干重复的数，找到其中一个
	 */
	// Parameters:
	// numbers: an array of integers
	// length: the length of array numbers
	// duplication: (Output) the duplicated number
	// in the array number,length of duplication array is 1,
	// so using duplication[0] = ? in implementation;
	// Here duplication like pointor in C/C++,
	// duplication[0] equal *duplication in C/C++
	// 这里要特别注意~返回任意重复的一个，赋值duplication[0]
	// Return value: true if the input is valid,
	// and there are some duplications in the array number
	// otherwise false
	public boolean duplicate(int numbers[], int length, int[] duplication) {
		boolean flag[] = new boolean[length];
		for (int i = 0; i < length; i++) {
			if (flag[numbers[i]]) {
				duplication[0] = numbers[i];
				return true;
			}
			flag[numbers[i]] = true;
		}
		return false;
	}

	/*
	 * 43.给定一个数组A[0,1,...,n-1],请构建一个数组B[0,1,...,n-1],
	 * 其中B中的元素B[i]=A[0]*A[1]*...*A[i-1]*A[i+1]*...*A[n-1]。 不能使用除法。 思想：从两边向中间乘
	 */
	public int[] multiply(int[] A) {
		int n = A.length;
		if (n == 0)
			return A;
		int B[] = new int[n];
		int i, ret;
		ret = 1;
		for (i = 0; i < n; i++) {
			B[i] = ret;
			ret *= A[i];
		}
		ret = 1;
		for (i = n - 1; i >= 0; i--) {
			B[i] *= ret;
			ret *= A[i];
		}
		return B;
	}

	/*
	 * 44.给定一棵二叉搜索树， 请找出其中的第k小的结点。 例如， （5，3，7，2，4，6，8） 中， 按结点数值大小顺序第三小结点的值为4。
	 */
	int count;

	public TreeNode traverse(TreeNode root) {
		if (root == null)
			return null;
		TreeNode res = null;
		res = traverse(root.left);
		count--;
		if (count == 0)
			return root;
		if (res != null)
			return res;
		return traverse(root.right);
	}

	TreeNode KthNode(TreeNode pRoot, int k) {
		count = k;
		return traverse(pRoot);
	}

	/*
	 * 44.从上到下按层打印二叉树， 同一层结点从左至右输出。 每一层输出一行。
	 */
	ArrayList<ArrayList<Integer>> Print(TreeNode pRoot) {
		ArrayList<ArrayList<Integer>> result = new ArrayList<>();
		ArrayList<TreeNode> current = new ArrayList<>();
		ArrayList<TreeNode> next;
		ArrayList<Integer> tmp;
		if (pRoot == null)
			return result;
		current.add(pRoot);
		while (current.size() > 0) {
			next = new ArrayList<>();
			tmp = new ArrayList<>();
			for (TreeNode t : current) {
				tmp.add(t.val);
				if (t.left != null)
					next.add(t.left);
				if (t.right != null)
					next.add(t.right);
			}
			result.add(tmp);
			current = next;
		}
		return result;
	}

	/*
	 * 45.给定一个数组和滑动窗口的大小，找出所有滑动窗口里数值的最大值。 即如果有k个滑动窗口，则有k个最大值 ·
	 */
	public ArrayList<Integer> maxInWindows(int[] num, int size) {
		ArrayList<Integer> res = new ArrayList<>();
		int i, j, max, n = num.length;
		if (n == 0 || size == 0 || n < size)
			return res;
		for (i = 0; i < n - size + 1; i++) {
			max = num[i];
			for (j = i; j < i + size; j++)
				if (num[j] > max)
					max = num[j];
			res.add(max);
		}
		return res;
	}

	public class TreeLinkNode {
		int val;
		TreeLinkNode left = null;
		TreeLinkNode right = null;
		TreeLinkNode next = null;

		TreeLinkNode(int val) {
			this.val = val;
		}
	}

	/*
	 * 46.二叉树节点的中序遍历的后继节点 思想， 中序后继：右子树的最左节点， 往父节点走时，第一次左拐 祖父节点的左边
	 */
	public TreeLinkNode GetNext(TreeLinkNode pNode) {
		if (pNode == null)
			return null;
		if (pNode.right == null) {
			while (pNode.next != null && pNode.next.right == pNode)
				pNode = pNode.next;
			return pNode.next;
		}
		pNode = pNode.right;
		while (pNode.left != null)
			pNode = pNode.left;
		return pNode;
	}

	/*
	 * 47.去掉链表中重复的节点 在一个排序的链表中，存在重复的结点， 请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。
	 */
	public ListNode deleteDuplication(ListNode pHead) {
		if (pHead == null || pHead.next == null)
			return pHead;
		ListNode head = new ListNode(0);
		head.next = pHead;
		ListNode tail = head, p, q;
		p = pHead;
		q = p.next;
		while (q != null) {
			if (p.val == q.val) {
				while (q != null && q.val == p.val)
					q = q.next;
				p = q;
				if (q == null)
					break;
				q = q.next;
			} else {
				tail.next = p;
				tail = tail.next;
				p = p.next;
				if (p == null)
					break;
				q = p.next;
			}
		}
		tail.next = p;
		return head.next;
	}

	/*
	 * 48.判断是否对称二叉树 方法：先生成镜像二叉树，在判断是否相同 非递归：DFS，用stack来保存成对的节点， 出栈的时候是成对出现的
	 * 入栈也是成对出现的
	 * 
	 * BFS，Queue保存节点，入队是成队的，出队也是成对的
	 */
	TreeNode Mirrors(TreeNode root) {
		if (root == null)
			return null;
		TreeNode nroot = new TreeNode(root.val);
		nroot.left = Mirrors(root.right);
		nroot.right = Mirrors(root.left);
		return nroot;
	}

	boolean isSame(TreeNode aroot, TreeNode broot) {
		if (aroot == null && broot == null)
			return true;
		if (aroot != null && broot != null) {
			if (aroot.val != broot.val)
				return false;
			return isSame(aroot.left, broot.left) && isSame(aroot.right, broot.right);
		}
		return false;
	}

	/* 方法2，直接判断自己和自己是否是镜像的 */
	boolean isSymmetrical(TreeNode pRoot, TreeNode qRoot) {
		if (pRoot == null && qRoot == null)
			return true;
		if (pRoot != null && qRoot != null && pRoot.val == qRoot.val)
			return isSymmetrical(pRoot.right, qRoot.left) && isSymmetrical(pRoot.left, qRoot.right);
		return false;
	}

	boolean isSymmetrical(TreeNode pRoot) {
		TreeNode miroot = Mirrors(pRoot);
		return isSame(miroot, pRoot);
	}

	/*
	 * 49.按之字型打印二叉树 即第一行按照从左到右的顺序打印， 第二层按照从右至左的顺序打印， 第三行按照从左到右的顺序打印，其他行以此类推
	 */
	public ArrayList<ArrayList<Integer>> ZPrint(TreeNode pRoot) {
		ArrayList<TreeNode> next, cur = new ArrayList<>();
		ArrayList<Integer> tmp;
		ArrayList<ArrayList<Integer>> res = new ArrayList<>();
		boolean flag = true;
		if (pRoot == null)
			return res;
		cur.add(pRoot);
		while (!cur.isEmpty()) {
			next = new ArrayList<>();
			tmp = new ArrayList<>();
			int i, n = cur.size();
			TreeNode t;
			for (i = 0; i < n; i++) {
				t = cur.get(i);
				if (t.left != null)
					next.add(t.left);
				if (t.right != null)
					next.add(t.right);
				if (flag)
					tmp.add(cur.get(i).val);
				else
					tmp.add(cur.get(n - i - 1).val);
			}
			res.add(tmp);
			cur = next;
			flag = !flag;
		}
		return res;
	}

	/*
	 * 50.数据流中的中位数 方法，使用最大堆和最小堆 小的一半用最大堆存储 大的一半用最小堆存储
	 */
	PriorityQueue<Integer> low = new PriorityQueue<>();
	PriorityQueue<Integer> high = new PriorityQueue<Integer>(new Comparator<Integer>() {
		@Override
		public int compare(Integer o1, Integer o2) {
			return o2 - o1;
		}
	});
	int counts = 0;

	public void Insert(Integer num) {
		if (counts % 2 == 0) {
			low.offer(num);
			int filteredMaxNum = low.poll();
			high.offer(filteredMaxNum);
		} else {
			high.offer(num);
			int filteredMinNum = high.poll();
			low.offer(filteredMinNum);
		}
		counts++;
	}

	public Double GetMedian() {
		if (low.size() == high.size())
			return (double) (low.peek() + high.peek()) / 2;
		return (double) high.peek();
	}

	/* 51.字符串中第一个只出现1次的字符 */
	public int FirstNotRepeatingChar(String str) {
		int table[] = new int[128];
		for (int i = 0; i < str.length(); i++) {
			char ch = str.charAt(i);
			int x = (int) ch;
			table[x]++;
		}
		for (int i = 0; i < str.length(); i++) {
			char ch = str.charAt(i);
			int x = (int) ch;
			if (table[x] == 1)
				return i;
		}
		return -1;
	}

	/*
	 * 52.判断链表中是否有环，并返回环的第一个节点 用两个指针，一个快指针，一个慢指针，速度为2倍，当两个相遇时，说明链表有环，
	 * 然后让其中一个回到起点，连个同时走，相同时，得到入口
	 */
	public ListNode EntryNodeOfLoop(ListNode pHead) {
		if (pHead == null || pHead.next == null)
			return null;
		ListNode p1 = pHead, p2 = pHead;
		while (p2 != null && p2.next != null) {
			p1 = p1.next;
			p2 = p2.next.next;
			if (p1 == p2) {
				p1 = pHead;
				while (p1 != p2) {
					p1 = p1.next;
					p2 = p2.next;
				}
				if (p1 == p2)
					return p1;
			}
		}
		return null;
	}

	/*
	 * 53.请实现一个函数用来找出字符流中第一个只出现一次的字符。 例如，当从字符流中只读出前两个字符"go"时，
	 * 第一个只出现一次的字符是"g"。当从该字符流中读出前六个字符“google"时， 第一个只出现一次的字符是"l"。
	 */
	int hash[] = new int[256];
	char queue[] = new char[256];
	int front = 0, tail = 0;

	public void Insert(char ch) {
		if (hash[(int) ch] == 0) {
			queue[tail] = ch;
			tail = (tail + 1) % 256;
			hash[(int) ch] = 1;
		} else
			hash[(int) ch] = 2;
	}

	public char FirstAppearingOnce() {
		char ch;
		while (front != tail) {
			ch = queue[front];
			if (hash[(int) ch] == 1)
				return ch;
			front = (front + 1) % 256;
		}
		return '#';
	}

	/*
	 * 54.求数组中逆序对的个数 逆序对，即A[i]>A[j]，且i>j 思想：归并排序，和分治
	 */
	int cnt = 0;

	public void MergeSort(int[] array, int i, int j) {
		if (i == j)
			return;
		int mid = (i + j) / 2;
		MergeSort(array, i, mid);
		MergeSort(array, mid + 1, j);
		int merge[] = new int[j - i + 1];
		int k = 0, x = i, y = mid + 1;
		while (x <= mid && y <= j) {
			if (array[x] <= array[y])
				merge[k++] = array[x++];
			else {
				cnt += (mid - x + 1);
				if (cnt >= 1000000007)
					cnt %= 1000000007;
				merge[k++] = array[y++];
			}
		}
		while (x <= mid)
			merge[k++] = array[x++];
		for (x = 0; x < k; x++)
			array[x + i] = merge[x];
	}

	public int InversePairs(int[] array) {
		MergeSort(array, 0, array.length - 1);
		return cnt;
	}

	/*
	 * 55.输入两个链表，找出它们的第一个公共结点。 先求链表长度，找出公共长度后，同时移动 方法有很多： 最笨的：对于A链表每个节点，B链表遍历，判断是否相同
	 * 二：先算长度，长的链表先走一段 三：hashset
	 */
	public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {
		int cnt1 = 0, cnt2 = 0;
		ListNode p1 = pHead1, p2 = pHead2;
		while (p1 != null) {
			p1 = p1.next;
			cnt1++;
		}
		while (p2 != null) {
			p2 = p2.next;
			cnt2++;
		}
		int k = cnt1 - cnt2;
		while (k > 0) {
			pHead1 = pHead1.next;
			k--;
		}
		while (k < 0) {
			pHead2 = pHead2.next;
			k++;
		}
		while (pHead1 != pHead2) {
			pHead1 = pHead1.next;
			pHead2 = pHead2.next;
		}
		return pHead1;
	}

	/* 56.统计一个数字在排序数组中出现的次数 */
	public int GetNumberOfK(int[] array, int k) {
		if (array.length == 0)
			return 0;
		int i = 0, j = array.length - 1;
		int mid, t;
		int cnt = 0;
		while (i < j) {
			mid = (i + j) / 2;
			if (k < array[mid])
				j = mid;
			else if (array[mid] < k)
				i = mid + 1;
			else if (array[mid] == k) {
				cnt++;
				for (t = mid + 1; t <= j && array[t] == k; t++)
					cnt++;
				for (t = mid - 1; t >= i && array[t] == k; t--)
					cnt++;
				return cnt;
			}
		}
		if (array[i] == k)
			cnt++;
		return cnt;
	}

	/*
	 * 57.输入一个正整数数组，把数组里所有数字拼接起来排成一个数， 打印能拼接出的所有数字中最小的一个。 例如输入数组{3，32，321}，
	 * 则打印出这三个数字能排成的最小数字为321323。
	 */
	Comparator<String> comparator1 = new Comparator<String>() {
		@Override
		public int compare(String o1, String o2) {
			int i = 0, j = 0;
			while (i < o1.length() && j < o2.length()) {
				if (o1.charAt(i) != o2.charAt(j))
					return (int) (o1.charAt(i) - o2.charAt(j));
				i++;
				j++;
			}
			while (i < o1.length()) {
				if (o1.charAt(i) != o2.charAt(j - 1))
					return (int) (o1.charAt(i) - o2.charAt(j - 1));
				i++;
			}
			while (j < o2.length()) {
				if (o1.charAt(i - 1) != o2.charAt(j))
					return (int) (o1.charAt(i - 1) - o2.charAt(j));
				j++;
			}
			return 0;
		}
	};

	public String PrintMinNumber(int[] numbers) {
		String num[] = new String[numbers.length];
		int i;
		for (i = 0; i < numbers.length; i++)
			num[i] = "" + numbers[i];
		Arrays.sort(num, comparator1);
		String str = "";
		for (i = 0; i < num.length; i++)
			str += num[i];
		return str;
	}

	/*
	 * 58.把只包含质因子2、3和5的数称作丑数（Ugly Number）。 例如6、8都是丑数， 但14不是，因为它包含质因子7。
	 * 习惯上我们把1当做是第一个丑数。 求按从小到大的顺序的第N个丑数。 将2、3、5三个队列
	 */
	public int min(int a, int b) {
		return (a < b) ? a : b;
	}

	public int GetUglyNumber_Solution(int index) {
		if (index <= 0)
			return 0;
		int result[] = new int[index];
		int p2 = 0, p3 = 0, p5 = 0, cnt = 0, tmp;
		result[0] = 1;
		while (cnt < index - 1) {
			tmp = min(2 * result[p2], min(3 * result[p3], 5 * result[p5]));
			if (tmp == 2 * result[p2])
				p2++;
			if (tmp == 3 * result[p3])
				p3++;
			if (tmp == 5 * result[p5])
				p5++;
			result[++cnt] = tmp;
		}
		return result[index - 1];
	}

	String Serialize(TreeNode root) {
		if (root == null)
			return "#,";
		String s = root.val + ",";
		s = s + Serialize(root.left);
		s = s + Serialize(root.right);
		return s;
	}

	/*
	 * 59.序列化和反序列化二叉树 序列化方式：先序遍历序列 层序方式
	 */
	TreeNode Deserialize(String str) {
		String vals[] = str.split(",");
		Queue<String> queue = new LinkedList<>();
		for (String val : vals)
			queue.offer(val);
		return Deserialize(queue);
	}

	TreeNode Deserialize(Queue<String> queue) {
		String val = queue.poll();
		if (val.equals("#"))
			return null;
		TreeNode root = new TreeNode(Integer.valueOf(val));
		root.left = Deserialize(queue);
		root.right = Deserialize(queue);
		return root;
	}

	/*
	 * 60.求出1~13的整数中1出现的次数,并算出100~1300的整数中1出现的次数？ 方法：计算每一位中1出现的次数
	 */
	public int NumberOf1Between1AndN_Solution(int n) {
		if (n <= 0)
			return 0;
		int cnt = 0;
		for (long i = 1; i <= n; i *= 10) {
			long p = i * 10;
			cnt += (n / p) * i;
			long x = n % p;
			if (x / i > 1)
				cnt += i;
			else if (x / i == 1)
				cnt += x % i + 1;
		}
		return cnt;
	}

	/*
	 * 61.判断字符串是否表示数值(包括整数和小数) +100、5e2、-123、3.14、-1E-16 正例
	 * 12e、1a3.14、1.2.3、+-5、12e+4.3 反例
	 */
	public boolean isNumeric(char[] str) {
		int i = 0;
		int n = str.length;
		int en = 0, dn = 0, num = 0;
		if (str[0] == '+' || str[0] == '-')
			i++;
		while (i < n) {
			if ('0' <= str[i] && str[i] <= '9') {
				num++;
				i++;
			} else if (str[i] == '.') {
				if (dn > 0 || en > 0)
					return false;
				dn++;
				i++;
			} else if (str[i] == 'e' || str[i] == 'E') {
				if (en > 0)
					return false;
				en++;
				i++;
				if (i < n && (str[i] == '+' || str[i] == '-'))
					i++;
				if (i == n)
					return false;
			} else {
				return false;
			}
		}
		return true;
	}

	/*
	 * 62.请实现一个函数用来匹配包括'.'和'*'的正则表达式。 模式中的字符'.'表示任意一个字符，而'*'表示它前面的字符可以出现任意次（包含0次）。
	 * 在本题中，匹配是指字符串的所有字符匹配整个模式。 例如，字符串"aaa"与模式"a.a"和"ab*ac*a"匹配，
	 * 但是与"aa.a"和"ab*a"均不匹配
	 */
	public boolean match(char[] str, char[] pattern) {
		if (str == null || pattern == null)
			return false;
		return match(str, 0, str.length, pattern, 0, pattern.length);
	}

	public boolean match(char[] str, int i, int n1, char[] pattern, int j, int n2) {
		if (i == n1 && j == n2)// 两个字符串到到最后了
			return true;
		if (i != n1 && j == n2)
			return false;// 模式串结束，匹配串没有结束
		if (i == n1 && j != n2) {
			// 模式串没有结束，匹配串已经结束
			while (j != n2) {
				if (pattern[j] != '*' && (j + 1 >= n2 || pattern[j + 1] != '*'))
					return false;// 模式串出现了一个非*字符，且后面没有*
				j++;
			}
			return true;
		}
		if (j + 1 == n2) {
			if (pattern[j] == '.' || pattern[j] == str[i])
				return match(str, i + 1, n1, pattern, j + 1, n2);
			return false;
		}
		if ((pattern[j] == '.' || pattern[j] == str[i]) && pattern[j + 1] != '*')
			return match(str, i + 1, n1, pattern, j + 1, n2);// 下一个字符不是*，当前位置匹配
		if ((pattern[j] == '.' || pattern[j] == str[i]) && pattern[j + 1] == '*')
			return match(str, i, n1, pattern, j + 2, n2) || match(str, i + 1, n1, pattern, j, n2);
		if (pattern[j + 1] == '*')
			return match(str, i, n1, pattern, j + 2, n2);
		return false;
	}

	/*
	 * 63.回溯 请设计一个函数，用来判断在一个矩阵中是否存在一条包含某字符串所有字符的路径。
	 * 路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。
	 * 如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。 例如 a b c e s f c s a d e e 这样的3 X 4
	 * 矩阵中包含一条字符串"bcced"的路径， 但是矩阵中不包含"abcb"路径，
	 * 因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
	 */
	public boolean hasPath(char[] matrix, int rows, int cols, char[] str) {
		boolean visited[] = new boolean[matrix.length];
		int x, y;
		for (x = 0; x < rows; x++)
			for (y = 0; y < cols; y++)
				if (hasPath(matrix, rows, cols, x, y, str, 0, visited))
					return true;
		return false;
	}

	public boolean hasPath(char[] matrix, int rows, int cols, int x, int y, char[] str, int k, boolean[] visited) {
		int index = x * cols + y;
		if (x < 0 || x >= rows || y < 0 || y >= cols || visited[index] || str[k] != matrix[index])
			return false;
		if (k == str.length - 1)
			return true;
		visited[index] = true;
		if (hasPath(matrix, rows, cols, x + 1, y, str, k + 1, visited)
				|| hasPath(matrix, rows, cols, x - 1, y, str, k + 1, visited)
				|| hasPath(matrix, rows, cols, x, y + 1, str, k + 1, visited)
				|| hasPath(matrix, rows, cols, x, y - 1, str, k + 1, visited))
			return true;
		visited[index] = false;
		return false;
	}

	/*
	 * 64.回溯 地上有一个m行和n列的方格。 * 一个机器人从坐标0,0的格子开始移动， 每一次只能向左，右，上，下四个方向移动一格，
	 * 但是不能进入行坐标和列坐标的数位之和大于k的格子。 例如，当k为18时，机器人能够进入方格（35,37）， 因为3+5+3+7 =
	 * 18。但是，它不能进入方格（35,38）， 因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
	 */
	public int movingCount(int threshold, int rows, int cols) {
		boolean visitFlag[] = new boolean[rows * cols];
		return movingCount(threshold, rows, cols, 0, 0, visitFlag);
	}

	public int movingCount(int threshold, int rows, int cols, int x, int y, boolean[] visitFlag) {
		int index = x * cols + y;
		if (x < 0 || x >= rows || y < 0 || y >= cols || visitFlag[index] || SumofNum(x) + SumofNum(y) > threshold)
			return 0;
		visitFlag[index] = true;
		return movingCount(threshold, rows, cols, x + 1, y, visitFlag)
				+ movingCount(threshold, rows, cols, x - 1, y, visitFlag)
				+ movingCount(threshold, rows, cols, x, y + 1, visitFlag)
				+ movingCount(threshold, rows, cols, x, y - 1, visitFlag) + 1;
	}

	int SumofNum(int n) {
		int s = 0;
		while (n > 0) {
			s += (n % 10);
			n /= 10;
		}
		return s;
	}

	/*
	 * 65.输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。 假设输入的前序遍历和中序遍历的结果中都不含重复的数字. 思考：如何重建
	 * 首先在先序遍历中找到根节点，然后在中序遍历序列中找左右子树，然后找左右子树的根节点， 递归
	 */
	public TreeNode reConstructBinaryTree(int[] pre, int[] in) {
		if (pre.length == 0 || in.length == 0)
			return null;
		if (pre.length != in.length)
			return null;
		TreeNode root = new TreeNode(pre[0]);
		for (int i = 0; i < in.length; i++)
			if (pre[0] == in[i]) {
				root.left = reConstructBinaryTree(Arrays.copyOfRange(pre, 1, i + 1), Arrays.copyOfRange(in, 0, i));
				root.right = reConstructBinaryTree(Arrays.copyOfRange(pre, i + 1, pre.length),
						Arrays.copyOfRange(in, i + 1, in.length));
			}
		return root;
	}

	/*
	 * 66. 用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型
	 * 方法，在一个栈中入，出时，从一个栈倒腾到另一个栈，出栈，然后再倒腾一遍
	 */
	Stack<Integer> stack1 = new Stack<Integer>();
	Stack<Integer> stack2 = new Stack<Integer>();

	public void push(int node) {
		stack1.push(node);
	}

	public int pop() {
		while (!stack1.empty())
			stack2.push(stack1.pop());
		int n = stack2.pop();
		while (!stack2.empty())
			stack1.push(stack2.pop());
		return n;
	}

	public void preOrder(TreeNode root) {
		TreeNode p = root;
		Stack<TreeNode> stack = new Stack<TreeNode>();
		while (p != null || !stack.isEmpty()) {
			while (p != null) {
				System.out.print(p.val);
				stack.push(p);
				p = p.left;
			}
			if (!stack.isEmpty()) {
				p = stack.pop();
				p = p.left;
			}
		}
	}

	public void inOrder(TreeNode root) {
		TreeNode p = root;
		Stack<TreeNode> stack = new Stack<>();
		while (p != null || !stack.isEmpty()) {
			while (p != null) {
				stack.push(p);
				p = p.left;
			}
			if (!stack.isEmpty()) {
				p = stack.pop();
				System.out.println(p.val);
				p = p.left;
			}
		}
	}

	public void postOrder(TreeNode root) {
		TreeNode p = root;
		Stack<TreeNode> stack = new Stack<>();
		TreeNode pre = null;
		stack.push(p);
		while (!stack.isEmpty()) {
			p = stack.peek();
			if ((p.left == null && p.right == null) || (pre != null && (pre == p.left || pre == p.right))) {
				System.out.println(p.val);
				pre = p;
			}else {
				if(p.left!=null)
					stack.push(p.left);
				if(p.right!=null)
					stack.push(p.right);
			}
		}
	}
	/*525. Contiguous Array
	 * 最长连续子数组，数组中0和1的个数相等
	 * 方法：求和，如果B[i]==B[j] 则，A[i-j]之间的01数目相等
	 * */
	public int findMaxLength(int[] nums) {
        int arr[] = new int[2*nums.length+1];
        Arrays.fill(arr,-2);
        arr[nums.length] = -1;
        int count = 0;
        int max = 0;
        for(int i=0;i<nums.length;i++){
            count = count +(nums[i]==0?-1:1);
            if(arr[count+nums.length]>=-1){
                max = Math.max(max,i-arr[count+nums.length]);
            }else{
                arr[count+nums.length] = i;
            }
        }
        return max;
    }
	
	/*529. Minesweeper
	 * 模拟扫雷点击一个方块
	 * 如果这个方块周围有雷，则展示该方块周围的雷数
	 * 如果这个方块周围没有雷，则展示该方块周围8个块，并按照该方式继续展开
	 * 如果这个方块为M（雷），则直接将该块标记为x，结束
	 * */
	public char[][] updateBoard(char[][] board, int[] click) {
        int x = click[0],y = click[1];
        if(board[x][y]=='M'){
            board[x][y] = 'X';
            return board;
        }else{
            reveal(board,x,y,board.length,board[0].length);
        }
        return board;
    }
    public void reveal(char[][] board,int x,int y,int m,int n){
        if(x<0 || x>m-1 || y<0 || y>n-1)
            return;
        if(board[x][y]=='E'){
            int count = 0;
            if(x>0 && board[x-1][y]=='M')
                count++;
            if(x<m-1 && board[x+1][y]=='M')
                count++;
            if(y>0 && board[x][y-1]=='M')
                count++;
            if(y<n-1 && board[x][y+1]=='M')
                count++;
            if(x>0 && y>0 && board[x-1][y-1]=='M')
                count++;
            if(x>0 && y<n-1 && board[x-1][y+1]=='M')
                count++;
            if(x<m-1 && y>0 && board[x+1][y-1]=='M')
                count++;
            if(x<m-1 && y<n-1 && board[x+1][y+1]=='M')
                count++;
            if(count>0){
                board[x][y] = (char)('0'+count);
                return;
            }
            else{
                board[x][y] = 'B';
                reveal(board,x-1,y,m,n);
                reveal(board,x-1,y-1,m,n);
                reveal(board,x-1,y+1,m,n);
                reveal(board,x,y-1,m,n);
                reveal(board,x,y+1,m,n);
                reveal(board,x+1,y-1,m,n);
                reveal(board,x+1,y,m,n);
                reveal(board,x+1,y+1,m,n);
            }
        }
    }
	
	
	public static void main(String[] args) {
		// TODO Auto-generated method stub
		LeetCode2 test = new LeetCode2();
		int B[] = { 7, 3, 2, 1, 6, 9, 10, 4 };
		int C[] = { 1, 2, 3, 4, 5 };
		int D[] = { 4, 5, 3, 2, 1 };
		int E[] = { 0, 2, 4, 3, 6 };
		int FFF[] = { 1, 2, 3, 4, 5, 6, 7, 0 };

		System.out.println(test.PrintMinNumber(FFF));

		System.out.println(test.InversePairs(FFF));

		String spq = "google";
		for (int i = 0; i < spq.length(); i++) {
			test.Insert(spq.charAt(i));
			System.out.println(test.FirstAppearingOnce());
		}

		test.Insert(5);
		System.out.println(test.GetMedian());
		test.Insert(2);
		System.out.println(test.GetMedian());

		TreeNode qqq = test.new TreeNode(6);
		TreeNode ppp = test.new TreeNode(5);
		qqq.left = ppp;
		ppp = test.new TreeNode(7);
		qqq.right = ppp;
		TreeNode rrr = test.new TreeNode(8);
		rrr.left = qqq;
		qqq = test.new TreeNode(10);
		ppp = test.new TreeNode(9);
		qqq.left = ppp;
		ppp = test.new TreeNode(11);
		qqq.right = ppp;
		rrr.right = qqq;
		System.out.println(test.KthNode(rrr, 1).val);

		System.out.println(test.isContinuous(E));
		ArrayList<Integer> sps = test.GetLeastNumbers_Solution(B, 4);

		for (int xxx : sps)
			System.out.print(xxx + " ");
		System.out.println();
		ArrayList<String> pers = test.Permutation("aacd");
		for (String sss : pers)
			System.out.println(sss);

		System.out.println("123".substring(0) + "4 ");
		System.out.println("123".substring(1) + "4 ");
		System.out.println("123".substring(2) + "4 ");
		System.out.println("123".substring(3) + "4 ");
		RandomListNode heads[] = new RandomListNode[5];
		for (int i = 0; i < 5; i++)
			heads[i] = test.new RandomListNode(i);
		for (int i = 0; i < 4; i++)
			heads[i].next = heads[i + 1];
		heads[0].random = heads[3];
		heads[4].random = heads[0];
		RandomListNode rest = test.Clone(heads[0]);

		TreeNode root = test.new TreeNode(5);
		TreeNode ts = test.new TreeNode(4);
		root.left = ts;
		ts = test.new TreeNode(7);
		root.right = ts;
		ts = test.new TreeNode(10);
		ts.left = root;
		root = test.new TreeNode(12);
		ts.right = root;
		ArrayList<ArrayList<Integer>> ress = test.FindPath(ts, 22);
		for (ArrayList<Integer> pp : ress) {
			for (int xx : pp)
				System.out.print(xx + " ");
			System.out.println();
		}
		System.out.println(test.VerifySquenceOfBST(E));
		System.out.println(test.IsPopOrder(C, D));
		int array[][] = new int[4][4];
		array[0][0] = 0;
		array[0][1] = 2;
		array[0][2] = 6;
		array[0][3] = 10;
		array[1][0] = 2;
		array[1][1] = 4;
		array[1][2] = 6;
		array[1][3] = 12;
		array[2][0] = 6;
		array[2][1] = 8;
		array[2][2] = 10;
		array[2][3] = 14;
		array[3][0] = 10;
		array[3][1] = 11;
		array[3][2] = 13;
		array[3][3] = 17;
		ArrayList<Integer> res = test.printMatrix(array);
		for (int c : res)
			System.out.println(c);

		ListNode head = test.new ListNode(C[0]);
		ListNode p, head1;

		ListNode tail = head;
		for (int k = 1; k < C.length; k++) {
			tail.next = test.new ListNode(C[k]);
			tail = tail.next;
		}
		head1 = test.new ListNode(B[0]);
		tail = head1;
		for (int k = 1; k < B.length; k++) {
			tail.next = test.new ListNode(B[k]);
			tail = tail.next;
		}
		p = test.Merge(head, head1);
		while (p != null) {
			System.out.println(p.val);
			p = p.next;
		}

		System.out.println(test.FindKthToTail(head, 5).val);
		p = test.ReverseList(head);
		tail = p;
		while (p != null) {
			System.out.println(p.val);
			p = p.next;
		}

		test.reOrderArray(B);
		for (int j = 0; j < B.length; j++)
			System.out.println(B[j]);
		System.out.println(-3 % 2);
		System.out.println(test.RectCover(2));
		System.out.println(test.JumpFloor(2));
		System.out.println(test.Fibonacci(3));
		int A[] = { 10, 11, 1, 2, 3, 4, 5, 6, 7, 8 };
		System.out.println(test.minNumberInRotateArray(A));
		System.out.println(test.Find(10, array));
		System.out.println(test.NumberOf1(5));
		System.out.println(test.numberOfLeadingZeros0(10));
	}
}
