import { Autocomplete, Button, CloseButton, Group, Image, LoadingOverlay, Pill, Text, type AutocompleteProps } from "@mantine/core";
import { useState } from "react";
import { True } from "./components";
import { Dropzone, IMAGE_MIME_TYPE, type FileWithPath } from '@mantine/dropzone';
import { Image as ImageIcon } from "lucide-react";
import clsx from "clsx";
import { useDisclosure } from "@mantine/hooks";
import { allTags, type Tag } from "./lib";
import classes from './upload.module.css';

function App() {
  const [files, setFiles] = useState<FileWithPath[]>([]);
  const [visible, { open, close }] = useDisclosure(false);
  const [taged, setTaged] = useState<Tag[]>([])
  const [mini, setMini] = useState(false)

  const previews = files.map((file, index) => {
    const imageUrl = URL.createObjectURL(file);
    return <div key={index}>
      <Image fit="contain" className="absolute top-0 left-0" h="100%" src={imageUrl} onLoad={() => URL.revokeObjectURL(imageUrl)} />
    </div>;
  });
  return (
    <>
      <div className="flex items-center justify-between text-2xl text-white pb-4">
        <div>DeepDanBooru</div>
        <button onClick={()=>setMini(!mini)} className="shrink-0 size-8 border rounded text-xs">{mini ? "M" : "N"}</button>
      </div>
      <div className='flex flex-col gap-2 w-full' >
        <form className="w-full flex flex-col gap-4">
          <LoadingOverlay pos="fixed" visible={visible} zIndex={1000} overlayProps={{ radius: "sm", blur: 2 }} />
          <div className={clsx("aspect-[245/345]", mini? "h-[560px]" : "h-[128px]", "relative rounded-lg overflow-hidden dark:bg-[var(--mantine-color-dark-8)]")}>
            <Dropzone radius="md" className={clsx("size-full flex justify-center items-center", previews.length != 0 ? "opacity-0" : "")} maxFiles={1} multiple={false} accept={IMAGE_MIME_TYPE} onDrop={setFiles}>
              <True bool={previews.length == 0}>
                <Group justify="center" gap="xl" mih={220} style={{ pointerEvents: 'none' }}>
                  <Dropzone.Idle>
                    <ImageIcon size={52} color="var(--mantine-color-dimmed)" />
                  </Dropzone.Idle>
                  <Text ta="center">Drop image here</Text>
                </Group>
              </True>
            </Dropzone>
            <True bool={previews.length > 0}>
              <CloseButton onClick={() => setFiles([])} className="top-2 right-2 z-10" pos="absolute" size="lg" />
            </True>
            {previews}
          </div>
          <Button disabled={files.length == 0} color="blue" className="shrink-0" onClick={() => {
            open()
            const form = new FormData();
            form.append("image", files[0]);
            fetch('/predict', {
              method: 'POST',
              body: form
            }).then(response => response.json())
              .then((response: Record<string, any>) => {
                const set = new Set<Tag>()
                for (const name of response.tags) {
                  const r = allTags.find(a => a.name == name)
                  if (r) {
                    set.add(r)
                  }
                }
                setTaged(Array.from(set))
                close()
              })
              .catch(() => close())
          }}>Predict</Button>
          {taged.map((item, index) => {
            return item.name
          }).join(", ")}
          {/* <Pill.Group pb="sm">{taged.map((item, index) => (
            <Pill key={index}
              classNames={{
                root: classes['pill-' + item.name] ?? classes['pill-' + item.category]
              }}
              withRemoveButton onRemove={() => {
                setTaged(taged.filter(a => a.name != item.name))
              }}>
              {item.name}
            </Pill>
          ))}</Pill.Group> */}
        </form>
      </div>
    </>
  )
}

export default App
