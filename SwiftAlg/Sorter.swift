//
//  Sorter.swift
//  SwiftAlg
//
//  Created by yxj on 2021/9/8.
//

import Foundation
/**
 912. 排序数组: https://leetcode-cn.com/problems/sort-an-array/
 */

class Sorter {
    func sortArray(_ nums: [Int]) -> [Int] {
        var newNums = nums
        quickSort(&newNums, 0 , nums.count - 1)
//        quickSort(&newNums)
        return newNums
    }

    func quickSort(_ nums: inout [Int], _ start: Int, _ end: Int){
        if start >= end {
            return
        }
        let pivot = nums[start]
        var left = start
        var right = end;
        while left < right {
            while left < right && nums[right] >= pivot {
                right-=1
            }
            nums[left] = nums[right]

            while left < right && nums[left] <= pivot {
                left+=1
            }
            nums[right] = nums[left]
        }
        nums[left] = pivot
        self.quickSort(&nums, start, left-1)
        self.quickSort(&nums, left+1, end)
    }

    func quickSort(_ nums: inout [Int]) {
        //https://www.jianshu.com/p/e00e060aa56c
        func partition(_ nums: inout [Int], _ left: Int, _ right: Int) -> Int {
            let pivot = nums[right]
            //less 左边全部小于 pivot，右边大于 pivot
            var less = left
            for current in (less...right-1) {
                if nums[current] < pivot {
                    nums.swapAt(less, current)
                    less += 1
                }
            }
            nums.swapAt(less, right)
            return less
        }

        func robot(_ nums: inout [Int], _ left: Int, _ right: Int) {
            if left >= right {
                return
            }
            let i = partition(&nums, left, right)
            robot(&nums, left, i-1)
            robot(&nums, i+1, right)
        }

        robot(&nums, 0, nums.endIndex - 1)
    }
}

