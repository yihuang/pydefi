// SPDX-License-Identifier: MIT
pragma solidity ^0.8.24;

/**
 * @title TransientReentrancyGuard
 * @notice Stateless mixin providing a ``nonReentrant`` modifier backed by
 *         EIP-1153 transient storage.  The lock slot is chosen high enough
 *         in the address space that programs are very unlikely to hit it
 *         by accident, but a malicious program executing in the inheriting
 *         contract's context can still ``TSTORE`` it directly — treat the
 *         guard as defense-in-depth against external re-entry, not a hard
 *         sandbox.
 */
abstract contract TransientReentrancyGuard {
    /// @dev Transient slot used for the lock.  Picked far above the slots
    ///      a program is expected to read or write so it cannot collide
    ///      with normal program state.
    uint256 private constant REENTRANCY_LOCK_SLOT = 0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff00;

    /// @notice Thrown when a guarded entry point is invoked re-entrantly.
    error Reentrant();

    modifier nonReentrant() {
        assembly {
            if tload(REENTRANCY_LOCK_SLOT) {
                mstore(0x00, 0x12d0ad9f) // bytes4(keccak256("Reentrant()"))
                revert(0x1c, 0x04)
            }
            tstore(REENTRANCY_LOCK_SLOT, 1)
        }
        _;
        assembly {
            tstore(REENTRANCY_LOCK_SLOT, 0)
        }
    }
}
