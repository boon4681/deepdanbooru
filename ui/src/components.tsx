import type { ReactNode } from "react";

export interface BasicProps {
    children: ReactNode | ReactNode[];
}


export function True({ bool, children }: { bool: boolean } & BasicProps) {
    if (bool) return children
}


export function Header({ children }: BasicProps) {
    return <>
        <div className='text-xl font-bold mb-6'>Artcon {children}</div>
    </>
}