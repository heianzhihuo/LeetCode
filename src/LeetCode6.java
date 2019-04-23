import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Scanner;

public class LeetCode6 {

	private static int totalPrice(int categoryCount, int totalVolume, int totalWeight, int[] volume, int[] weight,
			int[] stock, int[] price, int[] itemType) {

		return 0;
	}

	public static void main(String[] args) {
		Scanner in = new Scanner(System.in);
		String[] line = in.nextLine().split(",");
//总共商品种类
		int categoryCount = Integer.valueOf(line[0]);
//快递体积
		int totalVolume = Integer.valueOf(line[1]);
//快递重量
		int totalWeight = Integer.valueOf(line[2]);

//物品体积
		int[] volume = new int[50];
//重量
		int[] weight = new int[50];
//件数
		int[] stock = new int[50];
//价格
		int[] price = new int[50];
//类型
		int[] itemType = new int[50];

		for (int i = 1; i <= categoryCount; i++) {
			line = in.nextLine().split(",");
			volume[i] = Integer.valueOf(line[0]);
			weight[i] = Integer.valueOf(line[1]);
			stock[i] = Integer.valueOf(line[2]);
			price[i] = Integer.valueOf(line[3]);
			itemType[i] = Integer.valueOf(line[4]);
		}

		in.close();

		System.out.println(totalPrice(categoryCount, totalVolume, totalWeight, volume, weight, stock, price, itemType));

	}

}
