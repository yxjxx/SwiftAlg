//
//  Extensions.swift
//  SwiftAlg
//
//  Created by yxj on 2021/9/7.
//

import Foundation
/**
 实现 str[i] 访问 swift 字符串
 */
extension String {
    fileprivate subscript (i: Int) -> Character {
        return self[self.index(self.startIndex, offsetBy: i)]
    }

    subscript (i: Int) -> String {
        return String(self[i] as Character)
    }

    subscript (r: Range<Int>) -> String {
        let start = index(startIndex, offsetBy: r.lowerBound)
        let end = index(startIndex, offsetBy: r.upperBound)
        return String(self[start..<end])
    }

    subscript (r: ClosedRange<Int>) -> String {
        let start = index(startIndex, offsetBy: r.lowerBound)
        let end = index(startIndex, offsetBy: r.upperBound)
        return String(self[start...end])
    }
}

public protocol Identifiable {
    associatedtype ID: Hashable
    var id: ID { get }
}

extension Identifiable where Self: AnyObject {
    public var id: ObjectIdentifier {
        return ObjectIdentifier(self)
    }
}
