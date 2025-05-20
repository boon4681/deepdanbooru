import raw from "./selected_tags.csv?raw"
export type Tag = { id: string, name: string, category: string }

const tags = new Set()

for (const a of raw.replace(/\r/g, "").split("\n").filter(a => a).map(a => a.split(",")).slice(1)) {
    // console.log(a)
    tags.add({
        id: a[0] + "",
        name: a[1] + "",
        category: Number(a[2]),
        count: Number(a[3] ?? "0")
    })
}

export const allTags = Array.from(tags) as Tag[]