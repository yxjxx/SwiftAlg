//
//  Solution.swift
//  SwiftAlg
//
//  Created by yxj on 2021/3/22.
//

import Foundation

struct NumAndIndex: Comparable {
    static func < (lhs: NumAndIndex, rhs: NumAndIndex) -> Bool {
        return lhs.num > rhs.num
    }
    var num: Int
    var index: Int
    init(_ num: Int, _ index: Int) {
        self.num = num
        self.index = index
    }

}

class KthLargest {
    var queue: PriorityQueue<Int>
    var k: Int
    init(_ k: Int, _ nums: [Int]) {
        self.k = k
        queue = PriorityQueue()
        for num in nums {
            let _ = add(num)
        }
    }

    func add(_ val: Int) -> Int {
        queue.add(val)
        if queue.size > k {
            let _ = queue.dequeue()
        }
        return queue.peek()!
    }
}

/**
 * Your KthLargest object will be instantiated and called as such:
 * let obj = KthLargest(k, nums)
 * let ret_1: Int = obj.add(val)
 */

public class Solution {
//    152. 乘积最大子数组
//    func maxProduct(_ nums: [Int]) -> Int {
//
//    }
    func isCompleteTree(_ root: TreeNode?) -> Bool {
        if (root == nil) {
            return true
        }
        let a: Repeated<String> = repeatElement("SwiftRocks", count: 3)
        var queue: Queue<TreeNode?> = Queue()
        var leafShow: Bool = false
        queue.enqueue(root!)
        while (!queue.isEmpty) {
            let node = queue.dequeue()!
            if (node == nil) {
                leafShow = true
            } else if node != nil {
                if leafShow {
                    return false
                }
                queue.enqueue(node!.left)
                queue.enqueue(node!.right)
            }
        }
        return true
    }

    func searchRange(_ nums: [Int], _ target: Int) -> [Int] {
        func robot(_ nums: [Int], _ target: Int, _ lower: Bool) -> Int {
            var res = nums.count
            var left = 0
            var right = nums.count - 1
            while left <= right {
                let middle = (left + right) / 2
                if target < nums[middle] ||
                    (lower && target <= nums[middle]){
                    right = middle - 1
                    res = middle
                } else {
                    left = middle + 1
                }
            }
            return res
        }
        let start = robot(nums, target, true)
        let end = robot(nums, target, false) - 1
        if start <= end &&
            end < nums.count &&
            nums[start] == target &&
            nums[end] == target {
            return [start, end]
        }
        return [-1, -1]
    }
    //动态规划 (a)b a，b为缩小了的问题规模。
    func generateParenthesis(_ n: Int) -> [String] {
        var cache:[[String]?] = Array(repeating: nil, count: 10)
        cache[0] = [""]
        cache[1] = ["()"]
        func robot(_ n: Int) -> [String] {
            if cache[n] != nil {
                return cache[n]!
            }
            var res: [String] = []
            for i in (0..<n) {
                for a in robot(i) {
                    for b in robot(n - i - 1) {
                        res.append("(" + a + ")" + b)
                    }
                }
            }
            cache[n] = res
            return res
        }
        return robot(n)
    }
    func letterCombinations(_ digits: String) -> [String] {
        var dict: [Character:[Character]] = [:]
        dict["2"] = ["a", "b", "c"]
        dict["3"] = ["d", "e", "f"]
        dict["4"] = ["g", "h", "i"]
        dict["5"] = ["j", "k", "l"]
        dict["6"] = ["m", "n", "o"]
        dict["7"] = ["p", "q", "r", "s"]
        dict["8"] = ["t", "u", "v"]
        dict["9"] = ["w", "x", "y", "z"]
        let a = Array(digits)
        func dfs(_ digits: [Character], _ depth: Int, _ stack: inout Stack<Character>, _ ans: inout [String]) {
            if depth == digits.count {
                ans.append(String(stack.toArray()))
                return
            }
            let letters = dict[digits[depth]]!
            for (_, c) in letters.enumerated() {
                stack.push(c)
                dfs(digits, depth + 1, &stack, &ans)
                let _ = stack.pop()
            }
        }
        func dfs(_ digits: [Character], _ depth: Int, _ str: inout String, _ ans: inout [String]) {
            if depth == digits.count {
                ans.append(str)
                return
            }
            let letters = dict[digits[depth]]!
            for (_, c) in letters.enumerated() {
                str.append(c)
                dfs(digits, depth + 1, &str, &ans)
                str.removeLast()
            }
        }

        var stack: Stack<Character> = Stack()
        var str = ""
        var ans: [String] = []
        if (a.count == 0) { return ans }
        dfs(a, 0, &str, &ans)
        return ans
    }
    func search(_ nums: [Int], _ target: Int) -> Int {
        func robot(_ nums: [Int], _ target: Int, _ low: Int, _ high: Int) -> Int {
            if low > high {
                return -1
            }
            if low == high {
                return nums[low] == target ? low : -1
            }
            var nextLow = low
            var nextHigh = high
            let middle = (low + high) / 2
            if target == nums[middle] {
                return middle
            } else if target > nums[middle] {
                let m = (middle + high) / 2
                let increase = (nums[middle] <= nums[m] && nums[m] <= nums[high])//是否递增数列
                if !increase || (increase && target <= nums[high]) {
                    nextLow = middle + 1
                } else {
                    nextHigh = middle - 1
                }
            } else {
                let m = (low + middle) / 2
                let increase = (nums[low] <= nums[m] && nums[m] <= nums[middle])
                if !increase || (increase && nums[low] <= target) {
                    nextHigh = middle - 1
                } else {
                    nextLow = middle + 1
                }
            }
            return robot(nums, target, nextLow, nextHigh)
        }
        return robot(nums, target, 0, nums.count - 1)
    }

    func findDisappearedNumbers(_ nums: [Int]) -> [Int] {
        //数组做 hash
        var nums = nums
        let count = nums.count
        for num in nums {
            let x = (num - 1)  % count
            nums[x] = nums[x] + count
        }
        var ans: [Int] = []
        for (index, num) in nums.enumerated() {
            if num <= count {
                ans.append(index + 1)
            }
        }
        return ans
    }

    func topKFrequent(_ nums: [Int], _ k: Int) -> [Int] {
        var dict: [Int:Int] = [:]
        for (_, num) in nums.enumerated() {
            dict[num] = (dict[num] ?? 0) + 1
        }
        var buckets : [[Int]] = Array(repeating: Array(), count: nums.count)
        for (key, value) in dict {
            buckets[value - 1].append(key)
        }
        var res:[Int] = []
        for arr in buckets.reversed() {
            for num in arr {
                res.append(num)
            }
            if res.count >= k {
                return res
            }
        }
        return res
    }

    func missingTwo(_ nums: [Int]) -> [Int] {
        var ans: [Int] = []
        var nums = nums
        nums.append(0)
        nums.append(0)
        //数值的范围为 1...nums.count+2
        for index in (0..<nums.count) {
            var num = nums[index]
            while num != index + 1 && num != 0 {
                nums.swapAt(index, num-1)
                num = nums[index]
                if num == 0 {
                    break
                }
            }
        }
        for (index, num) in nums.enumerated() {
            if num == 0 {
                ans.append(index+1)
            }
        }
        return ans
    }

    func fourSumCount(_ nums1: [Int], _ nums2: [Int], _ nums3: [Int], _ nums4: [Int]) -> Int {
        var dict: [Int:Int] = [:]
        var ans = 0
        for a in nums1 {
            for b in nums2 {
                dict[a+b] = (dict[a+b] ?? 0) + 1
            }
        }
        for c in nums3 {
            for d in nums4 {
                if dict[-c-d] != nil {
                    ans += dict[-c-d] ?? 0
                }
            }
        }
        return ans
    }

    func fourSum(_ nums: [Int], _ target: Int) -> [[Int]] {
        func twoSum(_ nums: [Int], _ target: Int, _ start: Int, _ end: Int) -> [(Int, Int)] {
            var start = start
            var end = end
            var res : [(Int, Int)] = []
            while (start < end) {
                let s = nums[start] + nums[end]
                if s == target {
                    res.append((start, end))
                    start += 1
                    while start < end && nums[start] == nums[start - 1] {
                        start += 1
                    }
                    end -= 1
                    while start < end && nums[end] == nums[end + 1] {
                        end -= 1
                    }
                } else if s < target {
                    start += 1
                } else {
                    end -= 1
                }
            }
            return res
        }

        var ans: [[Int]] = []
        if nums.count < 4 {
            return []
        }
        let nums = nums.sorted{$0 < $1}
        for i in (0..<nums.count - 3 ) {
            if i > 0 && nums[i] == nums[i-1] {
                continue
            }
            for j in (i+1..<nums.count - 2) {
                if j > i+1 && nums[j] == nums[j-1] {
                    continue
                }
                let a = twoSum(nums, target-nums[i]-nums[j], j+1, nums.count - 1)
                for r in a {
                    ans.append([nums[i], nums[j], nums[r.0], nums[r.1]])
                }
            }
        }
        return ans
    }

    func permute(_ nums: [Int]) -> [[Int]] {
        func dfs(_ nums: [Int], _ depth: Int, _ path: inout Stack<Int>, _ used: inout [Bool], _ ans: inout [[Int]]) {
            if depth == nums.count {//递归结束的条件，遍历到了叶子结点
                ans.append(path.toArray())
                return
            }
            for (index, num) in nums.enumerated() {
                if used[index] {
                    continue
                }
                //回溯法， dfs 前后执行相反的操作
                path.push(num)
                used[index] = true
                dfs(nums, depth + 1, &path, &used, &ans)
                let _ = path.pop()
                used[index] = false
            }
        }

        var ans: [[Int]] = []
        var path: Stack<Int> = Stack()
        var used: [Bool] = Array.init(repeating: false, count: nums.count)
        let depth = 0
        dfs(nums, depth, &path, &used, &ans)
        return ans
    }

    func lengthOfLIS(_ nums: [Int]) -> Int {
        if nums.count <= 1 {
            return nums.count
        }
        // tail 数组的定义：长度为 i + 1 的上升子序列的末尾最小是几
        var tails : [Int] = []
        tails.append(nums[0])
        for i in (1..<nums.count) {
            if nums[i] > tails.last! {
                tails.append(nums[i])
            } else {
                // 使用二分查找法，在有序数组 tail 中，找到第 1 个大于等于 nums[i] 的元素，尝试让那个元素更小
                var left = 0
                var right = tails.count - 1
                while (left < right) {
                    let middle = left + (right - left)/2
                    if tails[middle] < nums[i] {
                        left = middle + 1
                    } else {
                        right = middle
                    }
                }
                tails[left] = nums[i]
            }
        }
        return tails.count
    }
    func lengthOfLIS2(_ nums: [Int]) -> Int {
        if nums.count <= 0 {
            return 0
        }
        var ans = 1
        //dp[i] 表示以 num[i] 结尾的数组的最长子序列长度
        //dp[i] = max(dp[j]) + 1 其中 j < i, nums[i] > nums[j]
        var dp: [Int] = Array.init(repeating: 1, count: nums.count)
        for i in (1..<nums.count) {
            for j in (0..<i) {
                if nums[i] > nums[j] {
                    dp[i] = max(dp[i], dp[j] + 1)
                }
            }
            ans = max(ans, dp[i])
        }
        return ans
    }
    func increasingTriplet(_ nums: [Int]) -> Bool {
        if nums.count < 3 {
            return false
        }
        var small = Int.max
        var medium = Int.max
        for i in (0..<nums.count) {
            if nums[i] <= small {
                small = nums[i]
            } else if nums[i] <= medium {
                medium = nums[i]
            } else {
                return true
            }
        }
        return false
    }
    func maxProfit(_ prices: [Int]) -> Int {
        var ans = 0
        for i in (1..<prices.count) {
            ans += max(0, prices[i] - prices[i-1])
        }
        return ans
    }
    func maxArea(_ height: [Int]) -> Int {
        var ans = 0
        var left = 0
        var right = height.count - 1
        while left < right {
            let area = min(height[left], height[right]) * (right - left)
            ans = max(ans, area)
            if height[left] <= height[right] {
                left += 1
            } else {
                right -= 1
            }
        }
        return ans
    }

    func candy(_ ratings: [Int]) -> Int {
        if ratings.count < 2 {
            return ratings.count
        }
        var res: [Int] = Array.init(repeating: 1, count: ratings.count)
        for i in (1..<ratings.count) {
            if ratings[i] > ratings[i-1] {
                res[i] = res[i-1] + 1
            }
        }
        for i in (1..<ratings.count).reversed() {
            if ratings[i] < ratings[i-1] {
                res[i-1] = max(res[i-1], res[i] + 1)
            }
        }
        let total = res.reduce(0, +)
        return total
    }

    func findContentChildren(_ g: [Int], _ s: [Int]) -> Int {
        let g = g.sorted{$0 < $1}
        let s = s.sorted{$0 < $1}
        var child = 0
        var cookie = 0
        while child < g.count && cookie < s.count {
            if g[child] <= s[cookie] {
                child += 1
            }
            cookie += 1
        }
        return child
    }

    func maxSlidingWindow(_ nums: [Int], _ k: Int) -> [Int] {
        var ans:[Int] = []
        var right = 0
        let queue: PriorityQueue<NumAndIndex> = PriorityQueue()
        while right < k {
            queue.add(NumAndIndex(nums[right], right))
            right += 1
        }
        if let a = queue.peek()?.num {
            ans.append(a)
        }
        for i in (k..<nums.count) {
            queue.add(NumAndIndex(nums[i], i))
            while queue.peek()!.index <= i - k {
                let _ = queue.dequeue()
            }
            ans.append(queue.peek()!.num)
        }
        return ans
    }

    func minWindow(_ s: String, _ t: String) -> String {
        let sa = Array(s)
        var left = 0
        var right = 0
        var dict: [Character:Int] = [:]//[char:t比当前s子串 char 多出现的次数]
        for c in Array(t) {
            dict[c] = (dict[c] ?? 0) + 1
        }
        var ans = ""
        var minLen = Int.max
        var counter = t.count
        //当窗口包含 t 全部所需的字符后（counter == 0），如果能收缩，我们就收缩窗口直到得到最小窗口。
        while right <= sa.count-1 {//一直移动右指针到出现覆盖t开始移动左指针直到不能覆盖t
            if (dict[sa[right]] ?? 0) > 0 {
                counter -= 1
            }
            dict[sa[right]] = (dict[sa[right]] ?? 0) - 1
            right += 1
            while counter == 0 {
                if minLen > right - left {
                    minLen = right - left
                    ans = s[left..<right]
                }
                if (dict[sa[left]] ?? 0) == 0 { //移动左指针直到不包含t中的全部
                    counter += 1
                }
                dict[sa[left]]! += 1
                left += 1
            }
        }
        return ans
    }

    //至多出现k次的子串最大长度
    func longestSubstring(_ s: String, _ k: Int) -> Int {
        let sa = Array(s)
        var left = 0
        var right = 0
        var dict: [Character:Int] = [:]//[char:times]
        var maxLen = 0
        var counter = 0
        while right <= sa.count-1 {
            if (dict[sa[right]] ?? 0) == 0 {
                counter += 1
            }
            dict[sa[right]] = (dict[sa[right]] ?? 0) + 1
            right += 1
            while counter > k {
                if (dict[sa[left]] ?? 0) == 1 {
                    counter -= 1
                }
                dict[sa[left]]! -= 1
                left += 1
            }
            maxLen = max(maxLen, right - left)
        }
        return maxLen
    }
    func lengthOfLongestSubstring(_ s: String) -> Int {
        var dict: [Character:Int] = [:]//[char:lastShowIndex]
        let sa = Array(s)
        var left = 0
        var right = 0
        var maxLen = 0
        while right <= sa.count-1 {
            let key = sa[right]
            if dict[key] != nil {//出现重复时开始移动左指针直到没有重复
                let pos = dict[key]!
                while left <= pos {
                    dict.removeValue(forKey: sa[left])
                    left += 1
                }
            }
            dict[key] = right
            maxLen = max(maxLen, right+1-left)
            right += 1
        }
        return maxLen
    }

    func lengthOfLongestSubstring3(_ s: String) -> Int {
        let sa = Array(s)
        var left = 0
        var right = 0
        var dict: [Character:Int] = [:]//[char:times]
        var maxLen = 0
        var counter = 0
        while right <= sa.count-1 {
            if (dict[sa[right]] ?? 0) > 0 {
                counter += 1 //一直移动右指针直到出现重复
            }
            dict[sa[right]] = (dict[sa[right]] ?? 0) + 1
            right += 1
            while counter > 0 {
                if (dict[sa[left]] ?? 0) > 1 {
                    counter -= 1//一直移动左指针直到没有重复了
                }
                dict[sa[left]]! -= 1
                left += 1
            }
            maxLen = max(maxLen, right - left)
        }
        return maxLen
    }

    func lengthOfLongestSubstring2(_ s: String) -> Int {
        let sa = Array(s)
        var left = 0
        var right = 0
        var dict: [Character:Int] = [:]//[char:times]
        var maxLen = 0
        var hasDuplicated = false
        while right <= sa.count-1 {
            if (dict[sa[right]] ?? 0) > 0 {
                hasDuplicated = true //一直移动右指针直到出现重复
            }
            dict[sa[right]] = (dict[sa[right]] ?? 0) + 1
            right += 1
            while hasDuplicated {
                if (dict[sa[left]] ?? 0) > 1 {
                    hasDuplicated = false//一直移动左指针直到没有重复了
                }
                dict[sa[left]]! -= 1
                left += 1
            }
            maxLen = max(maxLen, right - left)
        }
        return maxLen
    }
    /**
     中心发散，对于可能的最小回文中心 1个元素 / 2个相同元素
     往左右扩张，如果左右都相同继续扩展，直到左右不相同
     */
    func longestPalindrome(_ s: String) -> String {
        if s.count < 1 {
            return ""
        }
        let sa = Array(s)
        var start = 0
        var end = 0
        var maxLen = 1
        for (index, _) in sa.enumerated() {
            let len1 = expand(sa, index, index)
            let len2 = expand(sa, index, index + 1)
            let len = max(len1, len2)
            if len > maxLen {
                maxLen = len
                start = index - (maxLen-1)/2
                end = index + maxLen/2
            }
        }
        return s[start...end]
    }

    /**
     return 以 left 为左边界，right 为右边界，扩展开最长的回文串的长度
     */
    func expand(_ sa: [Character], _ left: Int, _ right: Int) -> Int {
        var left = left
        var right = right
        while left >= 0 && right <= sa.count - 1 && sa[left] == sa[right] {
            left -= 1
            right += 1
        }
        let len = right - left + 1
        return len - 2
    }
    /*
     动态规划
     一个元素肯定是回文，两个相同元素回文
     长度大于2的字符串左边界和右边界不同时肯定不回文，相同时等价于删除左右边界后缩小规模的字符串
     动态规划的顺序是从短的字符串到长的字符串，所以枚举子串长度，先求出长为2的全部子串是否回文，再求出长为3的全部子串是否回文，对于每一个回文的子字符串统计长度，更新最大长度，同时更新左边界
     结果即为 s[begin..<begin+max]
     */
    func longestPalindrome2(_ s: String) -> String {
        let len = s.count
        if len < 2 {
            return s
        }
        let sa = Array(s)
        var ansBegin = 0
        var max = 1 //初始最大长度为1，一个元素
        var dp: [[Bool?]] = Array(repeating: Array(repeating: nil, count: len), count: len)
        for i in (0..<s.count) {
            dp[i][i] = true//所有长度为1的子串都是回文的
        }
        //在状态转移方程中，我们是从长度较短的字符串向长度较长的字符串进行转移的，因此一定要注意动态规划的循环顺序。
        //先枚举子串长度
        for L in (2...len) {
            //枚举左边界
            for left in (0..<len) {//right - left + 1 = Len
                let right = L + left - 1//计算右边界
                if right >= len {
                    break
                }
                let subStrLen = right - left + 1
                if sa[left] != sa[right] {//对于任意子串左边界与右边界不等即不回文
                    dp[left][right] = false
                } else {
                    //子串左边界与右边界相等时，长度小于3即回文，长度大于3缩小规模再判断
                    if subStrLen <= 3 {
                        dp[left][right] = true
                    } else {
                        dp[left][right] = dp[left+1][right-1]
                    }
                }

                if dp[left][right]! && subStrLen > max {
                    max = subStrLen
                    ansBegin = left
                }
            }
        }

        return s[ansBegin..<ansBegin+max]
    }

    func intersection(_ nums1: [Int], _ nums2: [Int]) -> [Int] {
        var dict:[Int : Int] = [:]
        for num in nums1 {
            dict[num] = 1
        }
        var ans: Set<Int> = Set()
        for num in nums2 {
            if dict[num] != nil {
                ans.insert(num)
            }
        }
        return Array(ans)
    }
    func trailingZeroes(_ n: Int) -> Int {
        var ans = 0
        var n = n
        while n > 0 {
            n = n / 5
            ans += n
        }
        return ans
    }
    func containsDuplicate(_ nums: [Int]) -> Bool {
        var dict:[Int : Int] = [:]
        for n in nums {
            if dict[n] == nil {
                dict[n] = 1
            } else {
                dict[n] = -1
            }
        }
        for (_, v) in dict {
            if v == -1 {
                return true
            }
        }
        return false
    }
    func isPowerOfThree(_ n: Int) -> Bool {
        if n < 1 {
            return false
        }
        var n = n
        while n % 3 == 0 {
            n = n / 3
        }
        return (n == 1)
    }

    func firstUniqChar(_ s: String) -> Int {
        var dict:[Character : Int] = [:]
        for c in s {
            if dict[c] != nil {
                dict[c] = dict[c]! + 1
            } else {
                dict[c] = 1
            }
        }
        for (index, c) in s.enumerated() {
            if dict[c]! == 1 {
                return index
            }
        }
        return -1
    }

    func myPow(_ x: Double, _ n: Int) -> Double {
        if n == 0 {
            return 1
        }
        if n == 1 {
            return x
        }
        if n < 0 {
            let nn = -n
            return 1/myPow(x, nn)
        }
        let r = myPow(x, n/2)
        if n % 2 == 0 {
            return r*r
        } else {
            return Double(x)*r*r
        }
    }
    
    func getKthFromEnd(_ head: ListNode?, _ k: Int) -> ListNode? {
        if k < 1 {
            return nil
        }
        let dummy = ListNode()
        dummy.next = head
        var slow: ListNode? = dummy
        var fast: ListNode? = dummy
        var steps = k
        while steps > 0 {
            fast = fast?.next
            steps -= 1
        }
        while fast != nil {
            slow = slow?.next
            fast = fast?.next
        }
        return slow
    }

    func invertTree(_ root: TreeNode?) -> TreeNode? {
        if root == nil {
            return nil
        }
        let left = invertTree(root?.left)
        let right = invertTree(root?.right)
        root?.left = right
        root?.right = left
        return root
    }

    func climbStairs(_ n: Int) -> Int {
        var p = 0
        var q = 0
        var r = 1
        for _ in (1..<n) {
            p = q
            q = r
            r = p+q
        }
        return r
    }

    func isPalindrome(_ s: String) -> Bool {
        var count = 0
        var stack: Stack<String> = Stack()
        for c in s {
            if c.isNumber {
                stack.push(String(c))
                count += 1
            }
            if c.isLetter {
                stack.push(c.lowercased())
                count += 1
            }
        }
        for c in s {
            if c.isNumber || c.isLetter {
                let s = String(c).lowercased()
                if s != stack.pop() {
                    return false
                }
            }
        }
        return true
    }

    func merge(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        var current = m + n - 1
        var m = m-1
        var n = n-1
        while(m >= 0 && n >= 0) {
            if nums1[m] > nums2[n] {
                nums1[current] = nums1[m]
                m -= 1
            } else {
                nums1[current] = nums2[n]
                n -= 1
            }
            current -= 1
        }
        while n >= 0 {
            nums1[current] = nums2[n]
            n -= 1
            current -= 1
        }
    }

    func merge2(_ nums1: inout [Int], _ m: Int, _ nums2: [Int], _ n: Int) {
        if m == 0 {
            nums1 = nums2
            return
        }
        if n == 0 {
            return
        }
        var current = m + n - 1
        var m = m-1
        var n = n-1
        while (current >= 0) {
            if m >= 0 && n >= 0 {
                if nums1[m] >= nums2[n] {
                    nums1[current] = nums1[m]
                    m -= 1
                } else {
                    nums1[current] = nums2[n]
                    n -= 1
                }
            } else if m < 0 {
                nums1[current] = nums2[n]
            } else if n < 0 {
                nums1[current] = nums1[m]
            }
            current -= 1
        }
    }

    func mySqrt(_ x: Int) -> Int {
        if x == 0 {
            return 0
        } else if x < 4 {
            return 1
        } else if (x < 9) {
            return 2
        }
        var i = x/2
        while true {
            if x >= i*i && x < (i+1)*(i+1) {
                return i
            }
            i += 1
        }
    }
    func plusOne(_ digits: [Int]) -> [Int] {
        var ans: [Int] = digits
        var carry = 1
        let count = digits.count
        for (index, digit) in digits.reversed().enumerated() {
            ans[count - 1 - index] = (digit  + carry) % 10
            carry = (digit + carry) / 10
            if carry > 0 {
                continue
            } else {
                break
            }
        }
        if carry > 0 {
            ans.insert(carry, at: 0)
        }
        return ans
    }

    func strStr(_ haystack: String, _ needle: String) -> Int {
        let n = haystack.count
        let m = needle.count
        let harr = Array(haystack)
        let narr = Array(needle)
        if m == 0 {
            return 0
        }
        if n < m {
            return -1
        }
        for i in (0...n-m) {//对 haystack 中的每一位判断是否有子串于 needle 相等
            var flag = true
            for j in (0..<m) {
                if harr[i+j] != narr[j] {
                    flag = false
                    break
                }
            }
            if flag {
                return i
            }
        }
        return -1
    }

    func reverseString(_ s: inout [Character]) {
        let count = s.count
        var left = 0
        var right = count - 1
        while left < right {
            let temp = s[left]
            s[left] = s[right]
            s[right] = temp
            left = left + 1
            right = right - 1
        }
    }

    /**
     集合的全子集 数量肯定是 2的n次方（n为集合中元素的数量）
     对应是否选到集合中
     */
    func subsets(_ nums: [Int]) -> [[Int]] {
        var ans: [[Int]] = []
        let n = nums.count
        let p = (1 << n) - 1
        for i in (0...p) { //遍历每个子集
            var ele: [Int] = []
            for j in (0...p) { //对于每一个元素进行判断
                let q = (1 << j)
                if i & q != 0 { //判断当前位置是否需要放入这个元素
                    ele.append(nums[j])
                }
            }
            ans.append(ele)
        }
        return ans
    }

    /**
     给定一个包含 n + 1 个整数的数组 nums ，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。

     分析，nums 下标范围为 (0...n)，数字范围为 (1...n)，存在不同下标指向相同的值，
     会 val = nums[val] 会成环且不会回到 nums[0]
     */
    func findDuplicate(_ nums: [Int]) -> Int {
        var slow = 0
        var fast = 0
        repeat {
            slow = nums[slow]
            fast = nums[nums[fast]]
        } while (slow != fast)
        slow = 0
        while slow != fast {
            slow = nums[slow]
            fast = nums[fast]
        }
        return fast
    }

    func getSum(_ a: Int, _ b: Int) -> Int {
        var a = a
        var b = b
        while b != 0 {
            let carry = a & b
            a = a ^ b // 不进位加法
            b = carry << 1
        }
        return a
    }

    func reverseBits(_ n: Int) -> Int {
        var ans = 0
        var input = n
        for i in (0...31) {
            let lastBit = input & 1
            ans = ans | ( lastBit << (31 - i) )//逐位颠倒
            input = input >> 1
        }
        return ans
    }

    func hammingWeight(_ n: Int) -> Int {
        var ans = 0
        var input = n
        while input != 0 {
            input = input & (input - 1)
            ans += 1
        }
        return ans
    }

    func singleNumber(_ nums: [Int]) -> Int {
        let count = nums.count
        var res = nums[0]
        for index in (1..<count) {
            res = res ^ nums[index]
        }
        return res
    }

    /**
     对一个数进行两次完全相同的异或运算会得到原来的数
     */
    func missingNumber(_ nums: [Int]) -> Int {
        var ans = nums.count
        for index in (0..<nums.count) {
            ans = ans ^ index ^ nums[index]
        }
        return ans
    }
    func missingNumber2(_ nums: [Int]) -> Int {
        //求和有溢出风险
        let n = nums.count
        var sum = (1+n)*n/2
        for num in nums {
            sum -= num
        }
        return sum
    }

    func addTwoNumbers(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        if l1 == nil && l2 == nil {
            return nil
        }
        if l1 == nil || l2 == nil {
            return (l1 == nil) ? l2 : l1
        }
        var head: ListNode?
        var tail: ListNode?
        var addOn = 0//进位
        var l1 = l1
        var l2 = l2
        while l1 != nil || l2 != nil { //如果两个链表的长度不同，则可以认为长度短的链表的后面有若干个 0
            let val = (l1?.val ?? 0) + (l2?.val ?? 0) + addOn
            addOn = val / 10
            let node = ListNode(val % 10)
            if head == nil {
                head = node
                tail = node
            } else {
                tail?.next = node
                tail = tail?.next
            }
            l1 = l1?.next
            l2 = l2?.next
        }
        if addOn != 0 { //如果链表遍历结束后,还有进位还需要在答案链表的后面附加一个节点
            let node = ListNode(addOn)
            tail?.next = node
        }
        return head
    }
    /**
     因为需要删除结点，所以 slow 指向待删除结点的上一个结点
     */
    func removeNthFromEnd(_ head: ListNode?, _ n: Int) -> ListNode? {
        let preHead = ListNode()
        preHead.next = head
        var slow: ListNode? = preHead
        var fast: ListNode? = head
        for _ in (1...n) {//fast 指针先走 n 步
            fast = fast?.next
            if fast == nil {
                break
            }
        }
        while fast != nil { //fast 和 slow 同步移动，fast 指向 nil 时， slow 指向待删除结点的上一个结点
            fast = fast?.next
            slow = slow?.next
        }
        if slow?.next != nil  {
            slow?.next = slow?.next?.next
        }
        return preHead.next
    }

    func detectCycle(_ head: ListNode?) -> ListNode? {
        /**
         由数学特点发现，链表有环的时候, 快慢指针在环内相遇后，慢指针和头结点出发的指针同时单步移动会在环的入口处相遇
         */
        if head == nil || head?.next == nil {
            return nil//空链表或单个节点的链表肯定无环
        }
        var slow: ListNode? = head
        var fast: ListNode? = head
        while fast != nil {
            slow = slow?.next
            if fast?.next == nil {
                return nil
            }
            fast = fast?.next?.next
            if fast === slow {
                var ptr = head
                while ptr !== slow {
                    ptr = ptr?.next
                    slow = slow?.next
                }
                return ptr
            }
        }
        return nil
    }

    func hasCycle(_ head: ListNode?) -> Bool {
        /**
         链表时候有环的判定, 快慢指针是否相遇
         */
        if head == nil || head?.next == nil {
            return false//空链表或单个节点的链表肯定无环
        }
        var slow: ListNode? = head
        var fast: ListNode? = head?.next
        while slow !== fast {
            if fast == nil || fast?.next == nil {
                return false
            }
            slow = slow?.next
            fast = fast?.next?.next
        }
        return true
    }

    func deleteNode(_ node: ListNode?) {
        /**
         无法访问链表，只给定会尾结点的结点，删除改结点，只能将这个结点内容更新为它的下一个结点
         */
        node?.val = (node?.next!.val)!
        node?.next = node?.next?.next
    }

    func getIntersectionNode(_ headA: ListNode?, _ headB: ListNode?) -> ListNode? {
        /**
         listA length = a
         listB length = b
         common node length = c
         a + (b-c) = b + (a-c)
         */
        var A = headA
        var B = headB
        while A !== B {
            A = (A != nil) ? A?.next : headB
            B = (B != nil) ? B?.next : headA
        }
        return A
    }

    func mergeTwoLists(_ l1: ListNode?, _ l2: ListNode?) -> ListNode? {
        let preHead: ListNode? = ListNode(-1)
        var preV = preHead
        var p = l1
        var q = l2
        while p != nil && q != nil {
            if p!.val <= q!.val {
                preV?.next = p
                p = p?.next
            } else {
                preV?.next = q
                q = q?.next
            }
            preV = preV?.next
        }
        if p != nil {
            preV?.next = p
        }
        if q != nil {
            preV?.next = q
        }
        return preHead?.next
    }

    func longestCommonPrefix(_ strs: [String]) -> String {
        func myCharAtIdx(_ s: String, _ idx: Int) -> String {
            if idx > s.count - 1 {
                return ""
            }
            let i = s.index(s.startIndex, offsetBy: idx)
            return String(s[i])
        }
        if strs.count <= 0 {
            return ""
        }
        let length = strs[0].count
        let count = strs.count
        for i in (0..<length) {
            let c = myCharAtIdx(strs[0], i)
            for j in (1..<count) {
                if c != myCharAtIdx(strs[j], i) {
                    let s = strs[0]
                    let endIndex = s.index(s.startIndex, offsetBy: i)
                    return String(s[..<endIndex])
                }
            }
        }
        return strs[0]
    }

    func romanToInt(_ s: String) -> Int {
        func myCharAtIdx(_ s: String, _ idx: Int) -> String {
            if idx > s.count - 1 {
                return ""
            }
            let i = s.index(s.startIndex, offsetBy: idx)
            return String(s[i])
        }
        let dict = ["I":1,
                    "V":5,
                    "X":10,
                    "L":50,
                    "C":100,
                    "D":500,
                    "M":1000]
        var result = 0
        let length = s.count
        for index in (0...length-1) {
            let value = dict[myCharAtIdx(s, index)] ?? 0
            if index < length - 1 && value < dict[myCharAtIdx(s, index+1)] ?? 0 {
                result -= value
            } else {
                result += value
            }

        }
        return result
    }

    func reverse(_ x: Int) -> Int {
        let flag = x > 0 ? 1 : -1
        var x: Int = abs(x)
        var result = 0
        while x > 0 {
            result = result * 10 + (x % 10)
            x = x / 10
        }
        if (flag == -1 && result > Int32.max) || (flag == 1 && result > Int32.max - 1) {
            return 0
        }
        return result * flag
    }

    func sumNumbers(_ root: TreeNode?) -> Int {
        func robot(_ root: TreeNode?, _ preSum: Int) -> Int {
            if root == nil {
                return 0
            }
            let sum = preSum * 10 + root!.val
            if root!.left == nil && root!.right == nil {
                return sum
            }
            return robot(root?.left, sum) + robot(root?.right, sum)
        }
        return robot(root, 0)
    }

    func maxPathSum(_ root: TreeNode?) -> Int {
        var ans = Int.min
        @discardableResult func robot(_ root: TreeNode?) -> Int {
            if root == nil {
                return 0
            }
            let left = max(0, robot(root?.left))
            let right = max(0, robot(root?.right))
            let lmr = left + root!.val + right
            let ret = max(left, right) + root!.val
            ans = max(ans, max(lmr, ret))
            return ret
        }
        robot(root)
        return ans
    }

    func pathSum22(_ root: TreeNode?, _ targetSum: Int) -> Int {
        if root == nil {
            return 0
        }
        func robot(_ root: TreeNode?, _ targetSum: Int)  -> Int {
            var ans = 0
            if root == nil {
                return 0
            }
            let sum = targetSum - root!.val
            if sum == 0 {
                ans += 1
            }
            ans += robot(root?.left, sum)
            ans += robot(root?.right, sum)
            return ans
        }
        return robot(root, targetSum) + pathSum22(root?.left, targetSum) + pathSum22(root?.right, targetSum)
    }

    func pathSum(_ root: TreeNode?, _ targetSum: Int) -> [[Int]] {
        var ans: [[Int]] = []
        func dfs(_ root: TreeNode?, _ targetSum: Int, _ path: inout [Int]) {
            if root == nil {
                return
            }
            path.append(root!.val)
            if root?.left == nil && root?.right == nil && targetSum == root!.val {
                ans.append(path)
            }
            dfs(root?.left, targetSum - root!.val, &path)
            dfs(root?.right, targetSum - root!.val, &path)
            path.removeLast()
        }
        var path: [Int] = []
        dfs(root, targetSum, &path)
        return ans
    }

    func pathSum2(_ root: TreeNode?, _ targetSum: Int) -> [[Int]] {
        if root == nil {
            return []
        }
        var ans: [[Int]] = []
        var nodeQueue: Queue<TreeNode> = Queue()
        var valQueue: Queue<Int> = Queue()
        var pathQueue: Queue<[Int]> = Queue()
        nodeQueue.enqueue(root!)
        valQueue.enqueue(root!.val)
        pathQueue.enqueue([root!.val])
        while !nodeQueue.isEmpty {
            let node = nodeQueue.dequeue()
            let val = valQueue.dequeue()
            let path: [Int] = pathQueue.dequeue()!
            if node?.left == nil && node?.right == nil {
                if val == targetSum {
                    ans.append(path)
                } else {
                    continue
                }
            }
            if node?.left != nil {
                nodeQueue.enqueue(node!.left!)
                valQueue.enqueue(val! + node!.left!.val)
                var neoPath = path
                neoPath.append(node!.left!.val)
                pathQueue.enqueue(neoPath)
            }
            if node?.right != nil {
                nodeQueue.enqueue(node!.right!)
                valQueue.enqueue(val! + node!.right!.val)
                var neoPath = path
                neoPath.append(node!.right!.val)
                pathQueue.enqueue(neoPath)
            }
        }
        return ans
    }

    func hasPathSum(_ root: TreeNode?, _ targetSum: Int) -> Bool {
        if root == nil {
            return false
        }
        var nodeQueue: Queue<TreeNode> = Queue()
        var valQueue: Queue<Int> = Queue()
        nodeQueue.enqueue(root!)
        valQueue.enqueue(root!.val)
        while !nodeQueue.isEmpty {
            let node = nodeQueue.dequeue()
            let val = valQueue.dequeue()
            if node?.left == nil && node?.right == nil {
                if val == targetSum {
                    return true
                } else {
                    continue
                }
            }
            if node?.left != nil {
                nodeQueue.enqueue(node!.left!)
                valQueue.enqueue(val! + node!.left!.val)
            }
            if node?.right != nil {
                nodeQueue.enqueue(node!.right!)
                valQueue.enqueue(val! + node!.right!.val)
            }
        }
        return false
    }

    func hasPathSum2(_ root: TreeNode?, _ targetSum: Int) -> Bool {
        if root == nil {
            return false
        }
        if root?.left == nil && root?.right == nil && targetSum == root!.val {
            return true
        }
        let left = hasPathSum(root?.left, targetSum - root!.val)
        let right = hasPathSum(root?.right, targetSum - root!.val)
        return left || right
    }

    func sumOfLeftLeaves(_ root: TreeNode?) -> Int {
        var ans: Int = 0
        func dfs(_ root: TreeNode?, _ add: Int) {
            if root == nil {
                return
            }
            ans += add
            dfs(root?.left, (root?.left?.left == nil && root?.left?.right == nil) ? (root?.left?.val ?? 0) : 0)
            dfs(root?.right, 0)
        }
        dfs(root, 0)
        return ans
    }

    func lcaDeepestLeaves(_ root: TreeNode?) -> TreeNode? {
        var ans: TreeNode?
        func depth(_ root: TreeNode?) -> Int {
            if root == nil {
                return 0
            }
            let left = depth(root?.left)
            let right = depth(root?.right)
            return max(left, right) + 1
        }
        let left = depth(root?.left)
        let right = depth(root?.right)
        if left == right {
            return root //如果最深的叶子在左右子树都有的话，其公共父节点必须得是 root
        }
        if left > right {
            ans = lcaDeepestLeaves(root?.left)
        } else {
            ans = lcaDeepestLeaves(root?.right)
        }
        return ans
    }

    /**
     两个节点 p,q 分为两种情况：

     p 和 q 在相同子树中
     p 和 q 在不同子树中
     从根节点遍历，递归向左右子树查询节点信息
     递归终止条件：如果当前节点为空或等于 p 或 q，则返回当前节点

     递归遍历左右子树，如果左右子树查到节点都不为空，则表明 p 和 q 分别在左右子树中，因此，当前节点即为最近公共祖先；
     如果左右子树其中一个不为空，则返回非空节点。
    */
    func lowestCommonAncestor(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        if (root == nil || p === root || q === root) {
            return root
        }
        let left = lowestCommonAncestor(root?.left, p, q)
        let right = lowestCommonAncestor(root?.right, p, q)

        if left != nil && right != nil {
            return root
        }
        return (left == nil) ? right : left
    }

    func binaryTreePaths(_ root: TreeNode?) -> [String] {
        var ans:[String] = []
        if root == nil {
            return []
        }
        func dfs(_ root: TreeNode?, _ ans: inout [String], _ path: String) {
            if root == nil {
                return
            }
            if root?.left == nil && root?.right == nil {
                ans.append( path + "\(root!.val)" )
            }
            dfs(root?.left, &ans, path + "\(root!.val)->")
            dfs(root?.right, &ans, path + "\(root!.val)->")
        }
        dfs(root, &ans, "")
        return ans
    }

    func findTilt(_ root: TreeNode?) -> Int {
        var ans = 0
        @discardableResult func sumOfTree(_ root: TreeNode?) -> Int {
            if root == nil {
                return 0
            }
            let left = sumOfTree(root?.left)
            let right = sumOfTree(root?.right)
            ans += abs(left-right)
            return root!.val + left + right
        }
        sumOfTree(root)
        return ans
    }

    func diameterOfBinaryTree(_ root: TreeNode?) -> Int {
        var ans: Int = 0
        @discardableResult func depth(_ root: TreeNode?) -> Int {
            if root == nil{
                return 0
            }
            let left = depth(root?.left)
            let right = depth(root?.right)
            ans = max(ans, left + right)
            return max(left, right) + 1
        }
        depth(root)
        return ans
    }

    func widthOfBinaryTree(_ root: TreeNode?) -> Int {
        guard root != nil else {
            return 0
        }
        var stack = [TreeNode]()
        var lists: [Int] = [1]
        var maxLength = 1
        stack.append(root!)
        while !stack.isEmpty {
            let N = stack.count
            for _ in 0..<N {
                // 这里是removeFirst
                let node = stack.removeFirst()
                let index = lists.removeFirst()
                if let left = node.left {
                    stack.append(left)
                    lists.append(index &* 2)            // 加&是为了控制表示范围
                }
                if let right = node.right {
                    stack.append(right)
                    lists.append(index &* 2 &+ 1)
                }
            }
            if lists.count >= 2 {
                maxLength = max(maxLength, lists.last! &- lists.first! &+ 1)
            }
        }
        return maxLength
    }

    func widthOfBinaryTree2(_ root: TreeNode?) -> Int {
        if root == nil {
            return 0
        }
        var queue : Queue<(TreeNode?, Int, Int)> = Queue()
        queue.enqueue((root, 0, 0))
        var ans = 0, currentDepth = 0, left = 0
        while !queue.isEmpty {
            let count = queue.count
            for _ in 1...count {
                var depth: Int
                var position: Int
                var node: TreeNode?
                (node, depth, position) = queue.dequeue()!
                if node?.left != nil {
                    queue.enqueue((node?.left, depth + 1, position &* 2))
                }
                if node?.right != nil {
                    queue.enqueue((node?.right, depth + 1, position &* 2 + 1))
                }
                if currentDepth != depth { //记录每层的第一个为 left
                    currentDepth = depth
                    left = position
                }
                ans = max(ans, position - left + 1)
            }
        }
        return ans
    }

    func countNodes(_ root: TreeNode?) -> Int {
        if root == nil {
            return 0
        }
        var p = root
        var left = 0
        var right = 0
        while p != nil {
            p = p?.left
            left += 1
        }
        p = root
        while p != nil {
            p = p?.right
            right += 1
        }
        if left == right {//满二叉树
            return 1 << left - 1
        }
        return countNodes(root?.left) + countNodes(root?.right) + 1
    }

    func isBalanced(_ root: TreeNode?) -> Bool {
        if root == nil {
            return true
        }
        let left = maxDepth(root?.left)
        let right = maxDepth(root?.right)
        return abs(left - right) <= 1 && isBalanced(root?.left) && isBalanced(root?.right)
    }

    func minDepth(_ root: TreeNode?) -> Int {
        if root == nil {
            return 0
        }
        /*
            0
             \
              2
         这种最小高度算2
         */
        let left = minDepth(root?.left)
        let right = minDepth(root?.right)

        if left == 0 || right == 0 {
            return (left + right + 1)
        } else {
            return min(left, right) + 1
        }
    }

    func maxDepth(_ root: TreeNode?) -> Int {
        if root == nil {
            return 0
        }
        return max(maxDepth(root?.left), maxDepth(root?.right)) + 1
    }

    func maxDepth2(_ root: TreeNode?) -> Int {
        var ans = 0
        maxDepthRobot(root, 1, &ans)
        return ans
    }

    func maxDepthRobot(_ root: TreeNode?, _ level: Int, _ maxLevel: inout Int) {
        if root == nil {
            return
        }
        if level > maxLevel {
            maxLevel = level
        }
        maxDepthRobot(root?.left, level + 1, &maxLevel)
        maxDepthRobot(root?.right, level + 1, &maxLevel)
    }

    func isSymmetric(_ root: TreeNode?) -> Bool {
        isSymmetricRobot(root, root)
    }

    func isSymmetricRobot(_ root1: TreeNode?, _ root2: TreeNode?) -> Bool {
        if root1 == nil && root2 == nil {
            return true
        }
        if root1 == nil || root2 == nil {
            return false
        }
        if root1!.val == root2!.val {
            return isSymmetricRobot(root1?.left, root2?.right) &&
                   isSymmetricRobot(root1?.right, root2?.left)
        }
        return false
    }

    func isSameTree(_ p: TreeNode?, _ q: TreeNode?) -> Bool {
        if p == nil && q == nil {
            return true
        }
        if p == nil || q == nil{
            return false
        }
        if p!.val == q!.val {
            return (isSameTree(p!.left, q?.left) && isSameTree(p!.right, q?.right))
        }
        return false
    }

    func insertIntoBST(_ root: TreeNode?, _ val: Int) -> TreeNode? {
        if root == nil {
            return TreeNode(val)
        }
        if root!.val < val{
            root?.right = insertIntoBST(root?.right, val)
        }
        if root!.val > val {
            root?.left = insertIntoBST(root?.left, val)
        }
        return root
    }

    func searchBST(_ root: TreeNode?, _ val: Int) -> TreeNode? {
        if root == nil {
            return nil
        }
        if (root!.val < val)  {
            return searchBST(root!.right, val)
        } else if (root!.val > val)  {
            return searchBST(root!.left, val)
        } else {
            return root
        }
    }

    func lowestCommonAncestorSearch(_ root: TreeNode?, _ p: TreeNode?, _ q: TreeNode?) -> TreeNode? {
        if root == nil || p == nil || q == nil {
            return nil
        }
        var ancestor = root
        while true {
            if ancestor!.val > p!.val && ancestor!.val > q!.val  {
                ancestor = ancestor?.left
            } else if ancestor!.val < p!.val  && ancestor!.val < q!.val  {
                ancestor = ancestor?.right
            } else {
                break
            }
        }
        return ancestor
    }

    func kthSmallest(_ root: TreeNode?, _ k: Int) -> Int {
        var root = root
        var result : Int = Int.min
        var times = 0
        var stack : Stack<TreeNode> = Stack()
        while !stack.isEmpty || root != nil{
            while root != nil{
                stack.push(root!)
                root = root?.left
            }
            if !stack.isEmpty {
                let n : TreeNode = stack.pop()!
                times += 1
                result = n.val
                if times == k {
                    return result
                }
                root = n.right
            }
        }
        return result
    }

    func sortedArrayToBST(_ nums: [Int]) -> TreeNode? {
        return sortedArrayToBST(nums, 0, nums.count - 1)
    }

    func sortedArrayToBST(_ nums: [Int], _ start: Int, _ end:Int) -> TreeNode? {
        if start > end {
            return nil
        }
        if start == end {
            return TreeNode(nums[start])
        }
        if end - start == 1 {
            let right = TreeNode(nums[end])
            let root = TreeNode(nums[start], nil, right)
            return root
        }

        let middle = start + (end - start) / 2
        let left = sortedArrayToBST(nums, start, middle - 1)
        let right = sortedArrayToBST(nums, middle + 1, end)
        let root = TreeNode(nums[middle], left, right)
        return root
    }

    func trimBST(_ root: TreeNode?, _ low: Int, _ high: Int) -> TreeNode? {
        if root == nil {
            return nil
        }
        if root!.val < low {
            return trimBST(root?.right, low, high)
        }
        if root!.val > high {
            return trimBST(root?.left, low, high)
        }
//        if let left = root!.left {
//            if left.val < low {
//                root!.left = trimBST(left.right, low, high)
//            }
//        }
//        if let right = root!.right {
//            if right.val > high {
//                root!.right = trimBST(right.left, low, high)
//            }
//        }
        root?.left = trimBST(root?.left, low, high)
        root?.right = trimBST(root?.right, low, high)
        return root
    }

    func recoverTree(_ root: TreeNode?) {
        var root = root
        var node1: TreeNode?
        var node2: TreeNode?
        var pred: TreeNode?
        var stack : Stack<TreeNode> = Stack()
        while !stack.isEmpty || root != nil{
            while root != nil {
                stack.push(root!)
                root = root?.left
            }
            root = stack.pop()!
            if pred != nil && root!.val < pred!.val {
                node2 = root
                if node1 == nil {
                    node1 = pred
                } else {
                    break
                }
            }
            pred = root
            root = root?.right
        }
        swap(node1, node2)
    }

    func swap(_ node1: TreeNode?, _ node2: TreeNode?) {
        if node1 == nil || node2 == nil {
            return
        }
        let node1 = node1!
        let node2 = node2!
        let temp = node1.val
        node1.val = node2.val
        node2.val = temp
    }

    func isValidBST(_ root: TreeNode?) -> Bool {
        return isValidBSTRobot(root, Int.min, Int.max)
    }

    func isValidBSTRobot(_ root: TreeNode?, _ down: Int, _ up: Int) -> Bool {
        if root == nil {
            return true
        }
        let root = root!
        if root.val > down &&
           root.val < up &&
           isValidBSTRobot(root.left, down, min(root.val, up)) &&
           isValidBSTRobot(root.right, max(root.val, down), up){
            return true
        } else {
            return false
        }
    }

    func isValidBST2(_ root: TreeNode?) -> Bool {
        var root = root
        var currentLast : Int = Int.min
        var stack : Stack<TreeNode> = Stack()
        while !stack.isEmpty || root != nil{
            while root != nil{
                stack.push(root!)
                root = root?.left
            }
            if !stack.isEmpty {
                let n : TreeNode = stack.pop()!
                if currentLast >= n.val {
                    return false
                }
                currentLast = n.val
                root = n.right
            }
        }
        return true
    }

    func generateTrees(_ n: Int) -> [TreeNode?] {
        if n <= 0 {
            return [TreeNode]()
        }
        return generateTreesRobot(1, n)
    }

    func generateTreesRobot(_ start: Int, _ end: Int) -> [TreeNode?] {
        var allTrees: [TreeNode?] = []
        if start > end {
            allTrees.append(nil)
            return allTrees
        }
        for i in (start...end) {
            let leftTrees = generateTreesRobot(start, i-1)
            let rightTrees = generateTreesRobot(i+1, end)
            for leftNode in leftTrees {
                for rightNode in rightTrees {
                    let root = TreeNode(i, leftNode, rightNode)
                    allTrees.append(root)
                }
            }
        }
        return allTrees
    }

    func numTrees(_ n: Int) -> Int {
        var ans: [Int] = [1, 1]
        if n < ans.endIndex {
            return ans[n]
        }
        for i in (2...n) {
            var ansi: Int = 0
            for j in (1...i) {
                ansi += ans[j-1] * ans[i-j]
            }
            ans.append(ansi)
        }
        return ans[n]
    }

    func postorder(_ root: Node?) -> [Int] {
        var result: [Int] = []
        var stack : Stack<Node> = Stack()
        if root != nil{
            stack.push(root!)
        }
        while !stack.isEmpty {
            let n : Node = stack.pop()!
            result.insert(n.val, at: 0)
            for (_, node) in n.children.enumerated() {
                stack.push(node)
            }
        }
        return result
    }

    func postorder2(_ root: Node?) -> [Int] {
        var result: [Int] = []
        postorderRobot(root, &result)
        return result
    }

    func postorderRobot(_ root: Node?, _ result: inout [Int]) {
        if root == nil {
            return
        }
        for (_, node) in root!.children.enumerated() {
            postorderRobot(node, &result)
        }
        result.append(root!.val)//后序，从左到右再根
    }

    func preorder(_ root: Node?) -> [Int] {
        var result: [Int] = []
        if root == nil {
            return result
        }
        var stack: Stack<Node> = Stack()
        stack.push(root!)
        while !stack.isEmpty {
            let node = stack.pop()
            result.append(node!.val)
            for (_, n) in node!.children.reversed().enumerated() {
                stack.push(n)
            }
        }
        return result
    }

    func preorder2(_ root: Node?) -> [Int] {
        var result: [Int] = []
        preorderRobot(root, &result)
        return result
    }

    func preorderRobot(_ root: Node?, _ result: inout [Int]) {
        if root == nil {
            return
        }
        result.append(root!.val)
        for node in root!.children {
            preorderRobot(node, &result)
        }
    }

    func levelOrder(_ root: Node?) -> [[Int]] {
        var result: [[Int]] = []
        if root == nil {
            return result
        }
        var queue: Queue<Node> = Queue()
        queue.enqueue(root!)
        while !queue.isEmpty {
            var currentLevel: [Int] = []
            let currentLevelCount = queue.count
            for _ in Array(1...currentLevelCount) {
                let node = queue.dequeue()
                currentLevel.append(node!.val)
                for n in node!.children {
                    queue.enqueue(n)
                }
            }
            result.append(currentLevel)
        }
        return result
    }

    func zigzagLevelOrder(_ root: TreeNode?) -> [[Int]] {
        var result : [[Int]] = []
        if root == nil {
            return result
        }
        var q : Queue<TreeNode> = Queue()
        q.enqueue(root!)
        while !q.isEmpty {
            var currentLevel = [Int]()
            let count = q.count
            for _ in Array(1...count) {
                let n = q.dequeue()
                currentLevel.append(n!.val)
                if let left = n?.left {
                    q.enqueue(left)
                }
                if let right = n?.right {
                    q.enqueue(right)
                }
            }
            result.append(currentLevel)
        }
        for (index, array) in result.enumerated() {
            if index % 2 != 0 {
                result[index] = array.reversed()
            }
        }
        return result
    }

    func zigzagLevelOrder2(_ root: TreeNode?) -> [[Int]] {
        var result : [[Int]] = []
        if root == nil {
            return result
        }
        zigzagLevelOrderRobot(root, &result, 0)
        return result
    }

    func zigzagLevelOrderRobot(_ root: TreeNode?, _ result: inout [[Int]],  _ level: Int) {
        if root == nil {
            return
        }
        if result.count == level {
            result.append([Int]())
        }
        if level % 2 == 0 {
            result[level].append(root!.val)
        } else {
            result[level].insert(root!.val, at: 0)
        }

        zigzagLevelOrderRobot(root?.left, &result, level + 1)
        zigzagLevelOrderRobot(root?.right, &result, level + 1)
    }

    func verticalTraversal(_ root: TreeNode?) -> [[Int]] {
        var ans: [[Int]] = []
        var posNodes: [TreePositionNode] = []
        if root == nil {
            return []
        }
        verticalTraversalDfs(root, &posNodes, 0, 0)
        posNodes = posNodes.sorted { posNode1, posNode2 in
            if posNode1.column != posNode2.column {
                return (posNode1.column < posNode2.column)
            } else if posNode1.row != posNode2.row {
                return (posNode1.row < posNode2.row)
            } else {
                return (posNode1.val < posNode2.val)
            }
        }
        var lastColumn: Int = Int.min
        for posNode in posNodes {
            if lastColumn != posNode.column {
                ans.append([Int]())
            }
            ans[ans.count - 1].append(posNode.val)
            lastColumn = posNode.column
        }
        return ans
    }

    func verticalTraversalDfs(_ root: TreeNode?, _ posNodes: inout [TreePositionNode], _ column: Int, _ row: Int) {
        if root == nil {
            return
        }
        posNodes.append(TreePositionNode(root!.val, column, row))
        verticalTraversalDfs(root!.left, &posNodes, column - 1, row + 1)
        verticalTraversalDfs(root!.right, &posNodes, column + 1, row + 1)
    }

    func largestValues(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        if root == nil {
            return result
        }
        largestValuesRobot(root, &result, 0)
        return result
    }

    func largestValuesRobot(_ root: TreeNode?, _ result: inout [Int],  _ level: Int) {
        if root == nil {
            return
        }
        if result.count == level {
            result.append(root!.val)
        }
        if root!.val > result[level] {
            result[level] = root!.val
        }
        largestValuesRobot(root?.left, &result, level + 1)
        largestValuesRobot(root?.right, &result, level + 1)
    }

    func largestValues2(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        if root == nil {
            return result
        }
        var q : Queue<TreeNode> = Queue()
        q.enqueue(root!)
        while !q.isEmpty {
            let count = q.count
            var currentLevelMax: Int = Int.min
            for _ in Array(1...count) {
                let n = q.dequeue()
                if n!.val > currentLevelMax {
                    currentLevelMax = n!.val
                }
                if let left = n?.left {
                    q.enqueue(left)
                }
                if let right = n?.right {
                    q.enqueue(right)
                }
            }
            result.append(currentLevelMax)
        }
        return result
    }

    func rightSideView2(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        if root == nil {
            return result
        }
        var q : Queue<TreeNode> = Queue()
        q.enqueue(root!)
        while !q.isEmpty {
            let count = q.count
            for index in Array(1...count) {
                let n = q.dequeue()
                if index == count {
                    result.append(n!.val)
                }
                if let left = n?.left {
                    q.enqueue(left)
                }
                if let right = n?.right {
                    q.enqueue(right)
                }
            }
        }
        return result
    }

    func rightSideView(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        if root == nil {
            return result
        }
        robot(root, &result, 0)
        return result
    }

    func robot(_ root: TreeNode?, _ result: inout [Int],  _ level: Int) {
        if root == nil {
            return
        }
        if result.count == level {
            result.append(root!.val)
        }
        result[level] = root!.val
        robot(root?.left, &result, level + 1)
        robot(root?.right, &result, level + 1)
    }

    func levelOrderBottom(_ root: TreeNode?) -> [[Int]] {
        var result : [[Int]] = []
        if root == nil {
            return result
        }
        var q : Queue<TreeNode> = Queue()
        q.enqueue(root!)
        while !q.isEmpty {
            var currentLevel = [Int]()
            let count = q.count
            for _ in Array(1...count) {
                let n = q.dequeue()
                currentLevel.append(n!.val)
                if let left = n?.left {
                    q.enqueue(left)
                }
                if let right = n?.right {
                    q.enqueue(right)
                }
            }
            result.insert(currentLevel, at: 0)
        }
        return result
    }

    /*
     ans 为结果列表，level 为当前遍历的层数（初始为0）
     若 ans 的长度 = level，向 ans 增加一个空列表
     将节点值放入 ans 的第 level 个列表结尾
     遍历左右子节点，此时 level + 1
     */
    func levelOrder2(_ root: TreeNode?) -> [[Int]] {
        var result : [[Int]] = []
        if root == nil {
            return result
        }
        robot(root, &result, 0)
        return result
    }

    func robot(_ root: TreeNode?, _ result: inout [[Int]],  _ level: Int) {
        if root == nil {
            return
        }
        if result.count == level {
            result.append([Int]())
        }
        result[level].append(root!.val)
        robot(root?.left, &result, level + 1)
        robot(root?.right, &result, level + 1)
    }

    func levelOrder2x(_ root: TreeNode?) -> [[Int]] {
        var result : [[Int]] = []
        if root == nil {
            return result
        }
        var q : Queue<TreeNode> = Queue()
        q.enqueue(root!)
        while !q.isEmpty {
            var currentLevel = [Int]()
            let count = q.count
            for _ in Array(1...count) {
                let n = q.dequeue()
                currentLevel.append(n!.val)
                if let left = n?.left {
                    q.enqueue(left)
                }
                if let right = n?.right {
                    q.enqueue(right)
                }
            }
            result.append(currentLevel)
        }
        return result
    }

    /*
     根节点入栈，栈非空，栈顶出栈如果是第一次访问，入栈标记为已访问，再将其右子树入栈，左子树入栈。第二次访问直接出栈加入到结果数组中。
     */
func postorderTraversal(_ root: TreeNode?) -> [Int] {
    var result : [Int] = []
    var stack: Stack<StackNode> = Stack()
    if root == nil {
        return result
    }
    var node = StackNode(root)
    stack.push(node)
    while !stack.isEmpty {
        node = stack.pop()!
        if node.root == nil {
            continue
        }
        if !node.visit {
            node.visit = true
            stack.push(node)
            if let right = node.root?.right {
                stack.push(StackNode(right))
            }
            if let left = node.root?.left {
                stack.push(StackNode(left))
            }
        } else if (node.root != nil){
            result.append(node.root!.val)
        }
    }
    return result
}

    func postorderTraversal2(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        var stack : Stack<TreeNode> = Stack()
        if root != nil{
            stack.push(root!)
        }
        while !stack.isEmpty {
            let n : TreeNode = stack.pop()!
            result.insert(n.val, at: 0)
            if n.left != nil{
                stack.push(n.left!)
            }
            if n.right != nil{
                stack.push(n.right!)
            }
        }
        return result
    }
    
    func postorderTraversal_recursive(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        if root == nil {
            return []
        }
        result += self.postorderTraversal_recursive(root?.left)
        result += self.postorderTraversal_recursive(root?.right)
        result.append(root!.val)
        return result
    }
    
func inorderTraversal(_ root: TreeNode?) -> [Int] {
    var root = root
    var result : [Int] = []
    var stack : Stack<TreeNode> = Stack()
    while !stack.isEmpty || root != nil{
        while root != nil{
            stack.push(root!)
            root = root?.left
        }
        if !stack.isEmpty {
            let n : TreeNode = stack.pop()!
            result.append(n.val)
            root = n.right
        }
    }
    return result
}
    
    func inorderTraversal_recursive(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        if root == nil {
            return []
        }
        result += self.inorderTraversal_recursive(root?.left)
        result.append(root!.val)
        result += self.inorderTraversal_recursive(root?.right)
        return result
    }
    
    func preorderTraversal(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        var stack : Stack<TreeNode> = Stack()
        if root != nil{
            stack.push(root!)
        }
        while !stack.isEmpty {
            let n : TreeNode = stack.pop()!
            result.append(n.val)
            if n.right != nil{
                stack.push(n.right!)
            }
            if n.left != nil{
                stack.push(n.left!)
            }
        }
        return result
    }
    
    func preorderTraversal_recursive(_ root: TreeNode?) -> [Int] {
        var result : [Int] = []
        if root == nil {
            return []
        }
        result.append(root!.val)
        result += self.preorderTraversal_recursive(root?.left)
        result += self.preorderTraversal_recursive(root?.right)
        return result
    }

    func preorderTravel(_ root: TreeNode?) -> [Int] {
        var result: [Int] = []
        robot(root, &result)
        return result
    }

    func robot(_ root: TreeNode?, _ result: inout [Int]) {
        if root == nil {
            return
        }
        result.append(root!.val)
        robot(root!.left, &result)
        robot(root!.right, &result)
    }

    func maxSubArray(_ nums: [Int]) -> Int {
        var pre: Int = 0
        var result: Int = nums[0]
        for (_, value) in nums.enumerated() {
            pre = max(value, pre + value)
            result = max(result, pre)
        }
        return result
    }

    
    //[-1, 1, 2, 3, -4, 6, -1]
    //动态规划：递推关系f(n) = max(f(n-1) + A[n], A[n])
    //函数f(n)，表示以第n个数为结束点的子数列的最大和
    func maxSubArray2(_ nums: [Int]) -> Int {
        if nums.count <= 0 {
            return 0
        }
        var currentSum = nums[0]
        var result = nums[0]
        var start = 0
        var end = 0
        for (index, value) in nums.enumerated() {
            if (index == 0) {
                continue
            }
            if currentSum > 0 {
                currentSum += value
            } else {
                start = index
                end = index
                currentSum = value
            }
            
            if result < currentSum {
                result = currentSum
                end = index
            } else {
                
            }
        }
        print(nums[start...end])
        return result
    }
    
    func myCharAtIdx(_ s: String, _ idx: Int) -> Character {
        if idx > s.count - 1 {
            return Character("e")
        }
        let i = s.index(s.startIndex, offsetBy: idx)
        return s[i]
    }
    
    func myAtoi(_ s: String) -> Int {
        if s.count == 0 {
            return 0
        }
        var i = 0 //index
        var result = 0
        var flag = 1
        while (self.myCharAtIdx(s, i) == " ") {
            i+=1
        }
        if (self.myCharAtIdx(s, i) == "+") {
            i+=1
        } else if (self.myCharAtIdx(s, i) == "-") {
            flag = -1
            i+=1
        }
        
        while (i<s.count && self.myCharAtIdx(s, i).isNumber) {
            let c = self.myCharAtIdx(s, i)
            let item = c.asciiValue! - Character("0").asciiValue!
            if (result > Int32.max/10 || result == Int32.max/10 && item > 7) {
                return Int((flag == 1) ? Int32.max : Int32.min)
            }
            result = 10 * result + Int(item)        
            i+=1
        }
        return result*flag
    }
    
    
    func removeDuplicates(_ nums: inout [Int]) -> Int {
        if nums.count == 0 {
            return 0
        }
        var i = 0
        for j in Array(1..<nums.count) {
            let a = nums[i]
            let b = nums[j]
            if a != b {
                i+=1
                nums[i] = nums[j]
            }
        }
        let slice = nums[0...i]
        nums = Array(slice)
        return i+1
    }
    
    func reverseList(_ head: ListNode?) -> ListNode? {
        var currentNode = head
        var newNode : ListNode?
        while (currentNode != nil) {
            let tempNode = currentNode?.next
            currentNode?.next = newNode
            newNode = currentNode
            currentNode = tempNode
        }
        return newNode
    }
    
    func linkedListCount(_ head: ListNode?) -> Int {
        var i = 0
        var currentNode = head
        while currentNode != nil{
            i += 1
            currentNode = currentNode?.next
        }
        return i;
    }
    
    func isPalindrome(_ head: ListNode?) -> Bool {
        let cnt = self.linkedListCount(head)
        var middle = head
        for _ in Array(0..<cnt/2) {
            middle = middle?.next
        }
        if (cnt % 2 != 0) {
            middle = middle?.next
        }
        
        middle = self.reverseList(middle)
        var head = head
        for _ in Array(0..<cnt/2) {
            if head?.val == middle?.val {
                head = head?.next
                middle = middle?.next
            } else {
                return false
            }
        }
        return true
    }
    
    
    func quickSort(_ nums: inout [Int], _ start: Int, _ end: Int) -> Void {
        if start >= end {
            return
        }
        let pivot = nums[start]
        var left = start
        var right = end;
        while left < right {
            while left < right && nums[right] > pivot {
                right-=1
            }
            if left < right {
                nums[left] = nums[right]
                left+=1
            }
            
            while left < right && nums[left] < pivot {
                left+=1
            }
            if left < right {
                nums[right] = nums[left]
                right-=1
            }
        }
        nums[left] = pivot
        self.quickSort(&nums, start, left-1)
        self.quickSort(&nums, left+1, end)
    }
    
    func quickSortFirstK(_ nums: [Int], _ k: Int) -> [Int] {
        var subArr = Array(nums[0..<k])
        self.quickSort(&subArr, 0, k-1)
        return subArr
    }
    
    //节点 i 的左字节点下标2i+1， 右字节点下标2i+2
    func heapify(_ nums: inout [Int], _ idx: Int) -> Void {
        if (idx*2 + 1) > (nums.count-1) {
            return;
        }
        if nums[idx] > nums[2*idx+1] {
            nums.swapAt(2*idx+1, idx)
            self.heapify(&nums, 2*idx+1)
            if 2*idx+2 < nums.count-1 && nums[idx] > nums[2*idx+2] {
                nums.swapAt(2*idx+2, idx)
                self.heapify(&nums, 2*idx+2)
            }
        } else if 2*idx+2 < nums.count-1 && nums[idx] > nums[2*idx+2] {
            nums.swapAt(2*idx+2, idx)
            self.heapify(&nums, 2*idx+2)
        }
    }    
    
    public func findKthLargest(_ nums: inout [Int], _ k: Int) -> Int {
        var heap = self.quickSortFirstK(nums, k)
        for i in Array(k..<nums.count) {
            if nums[i] > heap[0] {
                heap[0] = nums[i]
                self.heapify(&heap, 0)
            }
        }
        return heap[0]
    }
}
